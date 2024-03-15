
#%% IP Adapter (face image, with CLIP image encoder)
class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class IPAdapterFull:
    def __init__(self, unet: UNet2DConditionModelInflated,
                 image_encoder_path,
                 ip_ckpt,
                 device,
                 lora_rank=128,
                 num_tokens=257, torch_dtype=torch.float32):
        self.unet = unet

        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.lora_rank = lora_rank
        self.torch_dtype = torch_dtype

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=self.torch_dtype
        )
        self.clip_image_processor = CLIPImageProcessor()

        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()


    def init_proj(self):  # 这里是IPAdapterFull的project直接抄来了
        image_proj_model = MLPProjModel(
            cross_attention_dim=768, #self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=self.torch_dtype)
        return image_proj_model

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"],
                                              strict=False)  # 因为没有to_k_lora
        print("IP Adapter CLIP face loaded")

    @torch.inference_mode()
    def set_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=weight_dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)

        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)

        self.image_prompt_embeds = image_prompt_embeds
        self.uncond_image_prompt_embeds = uncond_image_prompt_embeds
        # return image_prompt_embeds, uncond_image_prompt_embeds

    def add_clip_img_emb_to_txt_emb(self, txt_emb, nega_emb=False):

        if not nega_emb:
            prompt_embeds = torch.cat([txt_emb, self.image_prompt_embeds], dim=1)
        else:
            prompt_embeds = torch.cat([txt_emb, self.uncond_image_prompt_embeds], dim=1)

        return prompt_embeds

    def set_ip_scale(self, ip_scale):
        for i, proc_name in enumerate(self.unet.attn_processors):
            proc = self.unet.attn_processors[proc_name]
            if isinstance(proc, LoRAIPAttnProcessor):
                proc.set_ip_scale(ip_scale)
                if i == 15: print(f'ip_scale set to {ip_scale}')

image_encoder_path = f"{base_path}/IP-Adapter/models/image_encoder/"
ip_ckpt = f"{base_path}/IP-Adapter/models/ip-adapter-full-face_sd15.bin"

iPAdapterFull = IPAdapterFull(unet, image_encoder_path, ip_ckpt, device, num_tokens=257, torch_dtype=weight_dtype)

prompt_image = Image.open("/home/ids/xchen-21/FADING/avatar/prelim_results/reproduce_edits/reference_frame.png")
# prompt_image.resize((256, 256))
iPAdapterFull.set_image_embeds(prompt_image)
