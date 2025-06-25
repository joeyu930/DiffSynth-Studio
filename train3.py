import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
from pathlib import Path
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np

from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

torch.set_float32_matmul_precision('medium')

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        assert max_num_frames == num_frames, "max_num_frames should be equal to num_frames."
        assert frame_interval == 1, "frame_interval should be 1."
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            #v2.CenterCrop(size=(height, width)),
            #v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        
    def crop_and_resize(self, image):
        width, height = image.size
        #scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            #(round(height*scale), round(width*scale)),
            (self.height, self.width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < num_frames:
            reader.close()
            return None
        start_frame_id = (reader.count_frames() - num_frames)//2
        
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        #first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))

        return frames


    def load_video(self, file_path):
        frames = self.load_frames_using_imageio(file_path, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        path_pose = path.replace(".mp4", "_pose.mp4")
        video = self.load_video(path)
        video_pose = self.load_video(path_pose)
        data = {"text": text, "video": video, "path": path, "video_pose": video_pose}
        return data
    

    def __len__(self):
        return len(self.path)



class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, video_pose, path = batch["text"][0], batch["video"], batch["video_pose"], batch["path"][0]
        
        self.pipe.device = self.device
        pipe = self.pipe
        tiler_kwargs = self.tiler_kwargs
        if video is not None:
            assert video_pose is not None, "video_pose should not be None if video is not None."
            assert video.shape == video_pose.shape, "video and video_pose should have the same shape."
            print(f"video has shape {video.shape=}.")
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)            
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)
            vace_video = video_pose.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
            print(f"{vace_video.max()=}, {vace_video.min()=}")
            vace_mask = torch.ones_like(vace_video)
            inactive = vace_video * (1 - vace_mask) + 0 * vace_mask
            reactive = vace_video * vace_mask + 0 * (1 - vace_mask)
            inactive = pipe.encode_video(inactive, **tiler_kwargs)
            reactive = pipe.encode_video(reactive, **tiler_kwargs)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            vace_mask_latents = rearrange(vace_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            vace_reference_image_tensor = vace_video[:1, :, :1]
            vace_reference_latents0 = pipe.encode_video(vace_reference_image_tensor, **tiler_kwargs)

            vace_reference_latents = torch.concat((vace_reference_latents0, torch.zeros_like(vace_reference_latents0)), dim=1)
            vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
            print(f"{vace_mask_latents.shape=}, {vace_mask_latents.dtype=}, {vace_mask_latents.device=}")
            vace_mask_latents = vace_mask_latents.to(device=vace_video_latents.device, dtype=vace_video_latents.dtype)
            vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), 
                                              vace_mask_latents
                                              ), dim=2)
            
            # print the device of vace_reference_latents0 and latents
            print(f"{vace_reference_latents0.device=}, {latents.device=}")
            latents = torch.concat((vace_reference_latents0, latents), dim=2)            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)

            # image
            data = {"latents": latents[0], "prompt_emb_context": prompt_emb['context'][0].to(device=latents.device),
                    "vace_context": vace_context[0]}
            print(f"{latents.shape=}, {latents.dtype=}, {latents.device=}")
            print(f"{vace_context.shape=}, {vace_context.dtype=}, {vace_context.device=}")
            print(f"prompt_emb: {prompt_emb.keys()=}, {prompt_emb['context'].shape=}, {prompt_emb['context'].dtype=}, {prompt_emb['context'].device=}")
            torch.save(data, path + ".tensors.pth")


from pathlib import Path
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, steps_per_epoch=None):
        self.items = list(Path(base_path).glob("*.tensors.pth"))
        
        print(len(self.items), "items.")
        assert len(self.items) > 0
        if steps_per_epoch is None:
            self.steps_per_epoch = len(self.items)
        else:            
            if steps_per_epoch % len(self.items) != 0:
                print(f"Warning: steps_per_epoch {steps_per_epoch} is not a multiple of the number of items {len(self.items)}. "
                      "This may lead to repeated items in each epoch.")
                self.steps_per_epoch = (steps_per_epoch//len(self.items)+1) * len(self.items)
            else:
                self.steps_per_epoch = steps_per_epoch
        
        print(f"Dataset initialized with {len(self.items)} items, steps_per_epoch={self.steps_per_epoch}.")


    def __getitem__(self, index):
        path = self.items[index % len(self.items)]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data
    

    def __len__(self):
        return self.steps_per_epoch


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        #model_manager = ModelManager(torch_dtype=torch.float32, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            if 1:
                self.add_lora_to_model(
                self.pipe.vace,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
            self.pipe.vace.requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        self.pipe.vace.train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)
        prompt_context = batch["prompt_emb_context"].to(self.device)
        vace_context = batch["vace_context"].to(self.device)        

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        if 0: # original code   
            noise_pred = self.pipe.denoising_model()(
                noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
            )
        if 1: # new code
            dit = self.pipe.dit
            vace = self.pipe.vace
            vace_scale = 1.0

            # denoising step
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
            t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
            context = dit.text_embedding(prompt_context)
            x, (f, h, w) = dit.patchify(latents)
            freqs = torch.cat([
                dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
            vace_hints = vace(x, vace_context, context, t_mod, freqs)
            for block_id, block in enumerate(dit.blocks):
                x = block(x, context, t_mod, freqs)
                if vace_context is not None and block_id in vace.vace_layers_mapping:
                    x = x + vace_hints[vace.vace_layers_mapping[block_id]] * vace_scale
            x = dit.head(x, t)
            noise_pred = dit.unpatchify(x, (f, h, w))
            # end of denoising step

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch_size",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )
    trainer.test(model, dataloader)
    
    
def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
