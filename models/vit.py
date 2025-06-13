import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size  # H or W
        self.patch_size = patch_size
        self.grid_size_h = img_size // patch_size
        self.grid_size_w = img_size // patch_size
        self.num_patches = self.grid_size_h * self.grid_size_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: [B, C, H, W] or [B*T, C, H, W]
        x = self.proj(x)  # [B(*T), embed_dim, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)  # [B(*T), num_patches, embed_dim]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # For precise FAGTM, we need access to Q, K before softmax and averaging.
        # We'll use the main attn for standard forward, but compute FAGTM attention separately if needed.
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)  # Combined Q, K, V linear layer

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, compute_cls_attention_for_fagtm=False):

        if not compute_cls_attention_for_fagtm:
            residual = x
            x_norm = self.norm1(x)
            attended_x, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
            x = residual + attended_x
            residual = x
            x_norm = self.norm2(x)
            x = residual + self.mlp(x_norm)
            return x, None  # No attention map from this path

        # Path to compute CLS-to-patch attention for FAGTM (for the LAST block of teacher)
        # This manually implements the attention score calculation for FAGTM.
        B, N, C = x.shape  # N = 1 (CLS) + num_patches
        x_norm = self.norm1(x)

        qkv_projected = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_projected[0], qkv_projected[1], qkv_projected[2]  # q,k,v: [B, num_heads, N, head_dim]

        cls_q = q[:, :, 0:1, :]  # [B, num_heads, 1, head_dim] (CLS token query)
        patch_k = k[:, :, 1:, :]  # [B, num_heads, num_patches, head_dim] (Patch tokens keys)

        attn_scores = (cls_q @ patch_k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        attn_probs_cls_to_patches = attn_scores.softmax(dim=-1)

        avg_attn_probs_cls_to_patches = attn_probs_cls_to_patches.mean(dim=1)

        residual_main = x
        attended_x_main, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_main = residual_main + attended_x_main
        residual_main = x_main
        x_norm_main = self.norm2(x_main)
        x_main = residual_main + self.mlp(x_norm_main)

        return x_main, avg_attn_probs_cls_to_patches.squeeze(1)  # Return features and [B, num_patches] attention map

class MaskedReconstructionDecoder(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, num_output_channels=3, decoder_embed_dim=512, decoder_depth=1,
                 decoder_num_heads=16):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        num_patches = (224 // patch_size) ** 2
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=True)
        self.decoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=4.0)
            for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * num_output_channels, bias=True)

    def forward(self, x_encoded_tokens_all, bool_masked_pos):
        x_patch_tokens_encoded = x_encoded_tokens_all[:, 1:]
        x = self.decoder_embed(x_patch_tokens_encoded)
        B, N_total_patches, C_dec = x.shape
        mask_tokens_expanded = self.mask_token.expand(B, N_total_patches, -1)
        bool_masked_pos_expanded = bool_masked_pos.unsqueeze(-1).type_as(x)
        x_decode_input = (1 - bool_masked_pos_expanded) * x + bool_masked_pos_expanded * mask_tokens_expanded
        x_decode_input = x_decode_input + self.decoder_pos_embed[:, :N_total_patches, :]
        for blk in self.decoder_blocks:
            x_decode_input, _ = blk(x_decode_input)  # Block returns features, None for attention
        x_decode_input = self.decoder_norm(x_decode_input)
        reconstructed_pixels = self.decoder_pred(x_decode_input)
        return reconstructed_pixels


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 use_decoder=True, decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.use_decoder = use_decoder
        if use_decoder:
            self.decoder = MaskedReconstructionDecoder(embed_dim, patch_size, in_chans, decoder_embed_dim,
                                                       decoder_depth, decoder_num_heads)

    def forward_encoder_frames(self, x_frames, output_cls_attention_last_block=False):
        # x_frames: [B_frames, C, H, W] (e.g., B*T_g frames)
        B_frames = x_frames.shape[0]
        x_embedded = self.patch_embed(x_frames)  # [B_frames, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B_frames, -1, -1)  # [B_frames, 1, embed_dim]
        x_full = torch.cat((cls_tokens, x_embedded), dim=1)  # [B_frames, 1+num_patches, embed_dim]

        x_full = x_full + self.pos_embed
        x_full = self.pos_drop(x_full)

        cls_attention_map_last_block = None
        for i, blk in enumerate(self.blocks):
            is_last_block = (i == len(self.blocks) - 1)
            compute_fagtm_attn_here = is_last_block and output_cls_attention_last_block
            x_full, attn_map_if_computed = blk(x_full, compute_cls_attention_for_fagtm=compute_fagtm_attn_here)
            if compute_fagtm_attn_here:
                cls_attention_map_last_block = attn_map_if_computed  # [B_frames, num_patches]

        encoded_tokens = self.norm(x_full)  # [B_frames, 1+num_patches, embed_dim]
        cls_output = encoded_tokens[:, 0]  # [B_frames, embed_dim]

        if output_cls_attention_last_block:
            return cls_output, encoded_tokens, cls_attention_map_last_block
        return cls_output, encoded_tokens, None

    def forward(self, x, bool_masked_pos=None, is_teacher_for_fagtm=False, num_frames_in_clip=1):


        B_orig = x.shape[0]
        is_video_clip_input = x.ndim == 5  # (B, T, C, H, W)

        if is_video_clip_input:
            T_clip = x.shape[1]

            x_frames = x.reshape(B_orig * T_clip, x.shape[2], x.shape[3], x.shape[4])
        else:  # Single frame input (B, C, H, W)
            T_clip = 1
            x_frames = x

        if is_teacher_for_fagtm:
            cls_features_frames, _, cls_attention_map_frames = self.forward_encoder_frames(x_frames,
                                                                                           output_cls_attention_last_block=True)

            if is_video_clip_input:  # Reshape back if T_clip > 1
                cls_features_agg = cls_features_frames.view(B_orig, T_clip, -1).mean(dim=1)
                cls_attention_map_agg = cls_attention_map_frames.view(B_orig, T_clip, -1).mean(dim=1)
            else:  # Single frame view
                cls_features_agg = cls_features_frames
                cls_attention_map_agg = cls_attention_map_frames
            return cls_features_agg, cls_attention_map_agg

        # Standard student or teacher path (not for FAGTM attention map generation)
        cls_features_frames, all_encoded_tokens_frames, _ = self.forward_encoder_frames(x_frames,
                                                                                        output_cls_attention_last_block=False)

        if is_video_clip_input and T_clip > 1:
            # Aggregate CLS features over the temporal dimension for video clips
            # cls_features_frames is initially [B_orig * T_clip, embed_dim]
            # We reshape it to [B_orig, T_clip, embed_dim] and then take the mean over T_clip
            cls_features = cls_features_frames.view(B_orig, T_clip, -1).mean(dim=1)

            # all_encoded_tokens_frames is [B_orig * T_clip, 1 + num_patches, embed_dim].
            # For classification, when use_decoder is False (as in ClassificationModel),
            # this tensor is not used, so no further aggregation is strictly necessary here.
            all_encoded_tokens = all_encoded_tokens_frames
        else:  # Single frame input (B, C, H, W)
            cls_features = cls_features_frames
            all_encoded_tokens = all_encoded_tokens_frames

        if self.use_decoder and bool_masked_pos is not None:
            reconstructed_pixels = self.decoder(all_encoded_tokens, bool_masked_pos)
            return cls_features, reconstructed_pixels
        else:
            return cls_features