#!/usr/bin/env python
"""
SigLIP2 Network encoder for text-image similarity computation.

This module provides a SigLIP2-based text-image encoder that follows the same
interface as OpenCLIPNetwork, enabling evaluation of SigLIP2-extracted features.
Uses open_clip's SigLIP models which support both image and text encoding.
"""
import torch
import torchvision
import open_clip

class SigLIP2Network:
    """
    SigLIP2 text-image encoder for computing relevance between features and text prompts.

    This class provides the same interface as OpenCLIPNetwork but uses SigLIP2 model
    from open_clip, which outputs 768-dimensional embeddings.

    Attributes:
        device: torch device for computation
        clip_n_dims: embedding dimension (768 for SigLIP2)
        positives: list of positive text prompts
        negatives: list of negative text prompts
        pos_embeds: encoded positive prompt embeddings
        neg_embeds: encoded negative prompt embeddings
    """

    # Available SigLIP models from open_clip with 768-dim output
    MODEL_CONFIGS = {
        "siglip2_base_512": {
            "model_name": "ViT-B-16-SigLIP2-512",  # Base model, 512x512 input (default)
            "pretrained": "webli",
            "n_dims": 768,
            "image_size": 512,
        },
        "siglip2_large_512": {
            "model_name": "ViT-L-16-SigLIP2-512",  # Large model, 512x512 input
            "pretrained": "webli",
            "n_dims": 768,
            "image_size": 512,
        },
        "siglip2_base": {
            "model_name": "ViT-B-16-SigLIP2",  # Base model, 768-dim
            "pretrained": "webli",
            "n_dims": 768,
            "image_size": 224,
        },
        "siglip2_base_256": {
            "model_name": "ViT-B-16-SigLIP2-256",  # Base model, 256x256 input
            "pretrained": "webli",
            "n_dims": 768,
            "image_size": 256,
        },
        "siglip2_large_256": {
            "model_name": "ViT-L-16-SigLIP2-256",  # Large model, 256x256 input
            "pretrained": "webli",
            "n_dims": 768,
            "image_size": 256,
        },
        "siglip2_so400m": {
            "model_name": "ViT-SO400M-16-SigLIP2-256",  # SO400M model, 256x256 input
            "pretrained": "webli",
            "n_dims": 768,
            "image_size": 256,
        },
    }

    def __init__(self, device, model_variant="siglip2_base_512"):
        """
        Initialize SigLIP2 model and tokenizer from open_clip.

        Args:
            device: torch device (e.g., 'cuda:0', 'cuda:1', 'cpu')
            model_variant: which SigLIP2 model variant to use ('siglip2_base' or 'siglip2_large')
        """
        self.device = device
        config = self.MODEL_CONFIGS.get(model_variant, self.MODEL_CONFIGS["siglip2_base"])

        self.clip_n_dims = config["n_dims"]

        # Image preprocessing for SigLIP2
        self.process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((config["image_size"], config["image_size"]),
                                         interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5],  # SigLIP normalization
                std=[0.5, 0.5, 0.5],
            ),
        ])

        # Load SigLIP2 model from open_clip
        print(f"Loading SigLIP2 model from open_clip: {config['model_name']}")
        self.clip_model_type = config["model_name"]
        self.clip_model_pretrained = config["pretrained"]

        model, _, _ = open_clip.create_model_and_transforms(
            self.clip_model_type,
            pretrained=self.clip_model_pretrained,
            precision="fp16",
        )
        model.eval()
        self.model = model.to(device)

        self.tokenizer = open_clip.get_tokenizer(self.clip_model_type)

        # Default prompts (same as OpenCLIPNetwork)
        self.negatives = ("object", "things", "stuff", "texture")
        self.positives = (" ",)

        # Initialize with default prompts
        with torch.no_grad():
            self._encode_default_prompts()

        print(f"SigLIP2Network initialized with {self.clip_n_dims}-dim embeddings on {device}")

    def _encode_default_prompts(self):
        """Encode default positive and negative prompts."""
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(self.device)
            self.neg_embeds = self.model.encode_text(tok_phrases)

        self.pos_embeds = self.pos_embeds / self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds = self.neg_embeds / self.neg_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        """
        Compute relevance scores between embeddings and text prompts.

        Args:
            embed: feature embeddings of shape (n_pixels, embed_dim) - should be 768-dim
            positive_id: index of the positive phrase to compute relevance for

        Returns:
            relevance scores of shape (n_pixels, 2) containing positive and negative probabilities
        """
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(device=embed.device, dtype=embed.dtype)

        output = torch.mm(embed, p.T)  # (n_pixels, n_phrases)
        positive_vals = output[..., positive_id:positive_id + 1]  # (n_pixels, 1)
        negative_vals = output[..., len(self.positives):]  # (n_pixels, n_negatives)
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # (n_pixels, n_negatives)

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # (n_pixels, n_negatives, 2)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)

        return torch.gather(softmax, 1, best_id[..., None, None].expand(
            best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input, mask=None):
        """
        Encode an image into feature embeddings.

        Args:
            input: input image tensor
            mask: optional mask for region encoding

        Returns:
            image feature embeddings (768-dim)
        """
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input, mask=mask)

    def encode_text(self, text_list, device):
        """
        Encode text prompts into embeddings.

        Args:
            text_list: list of text strings
            device: target device

        Returns:
            text embeddings (768-dim)
        """
        text = self.tokenizer(text_list).to(device)
        return self.model.encode_text(text)

    def set_positives(self, text_list):
        """
        Set and encode positive text prompts.

        Args:
            text_list: list of positive text strings
        """
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat(
                [self.tokenizer(phrase) for phrase in self.positives]
                ).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds = self.pos_embeds / self.pos_embeds.norm(dim=-1, keepdim=True)

    def set_semantics(self, text_list):
        """
        Set and encode semantic labels.

        Args:
            text_list: list of semantic label strings
        """
        self.semantic_labels = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.semantic_labels]).to("cuda")
            self.semantic_embeds = self.model.encode_text(tok_phrases)
        self.semantic_embeds = self.semantic_embeds / self.semantic_embeds.norm(dim=-1, keepdim=True)

    def get_semantic_map(self, sem_map: torch.Tensor) -> torch.Tensor:
        """
        Generate semantic segmentation map from semantic feature maps.

        Args:
            sem_map: semantic feature maps of shape (n_levels, h, w, c) - c should be 768

        Returns:
            semantic prediction map of shape (n_levels, h, w)
        """
        n_levels, h, w, c = sem_map.shape
        pos_num = self.semantic_embeds.shape[0]
        phrases_embeds = torch.cat([self.semantic_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(sem_map.dtype)
        sem_pred = torch.zeros(n_levels, h, w)
        for i in range(n_levels):
            output = torch.mm(sem_map[i].view(-1, c), p.T)
            softmax = torch.softmax(10 * output, dim=-1)
            sem_pred[i] = torch.argmax(softmax, dim=-1).view(h, w)
            sem_pred[i][sem_pred[i] >= pos_num] = -1
        return sem_pred.long()

    def get_max_across(self, sem_map):
        """
        Process semantic map and return relevance maps for each granularity level.

        Args:
            sem_map: semantic feature maps of shape (n_levels, h, w, embed_dim) - embed_dim should be 768

        Returns:
            relevance maps of shape (n_levels, n_phrases, h, w)
        """
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]

        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1)  # (h*w, n_levels, embed_dim)

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j)
                pos_prob = probs[..., 0:1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)

        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map


class EncoderFactory:
    """
    Factory class for creating text-image encoders with automatic model selection.
    """

    @staticmethod
    def create_encoder(device, encoder_type="openclip", **kwargs):
        """
        Create a text-image encoder based on the specified type.

        Args:
            device: torch device
            encoder_type: type of encoder ('openclip' or 'siglip2')
            **kwargs: additional arguments passed to encoder constructor

        Returns:
            encoder instance (OpenCLIPNetwork or SigLIP2Network)
        """
        if encoder_type == "openclip":
            from openclip_encoder import OpenCLIPNetwork
            return OpenCLIPNetwork(device, **kwargs)
        elif encoder_type in ("siglip2", "siglip"):
            return SigLIP2Network(device, **kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}. "
                           f"Supported types: 'openclip', 'siglip2'")
