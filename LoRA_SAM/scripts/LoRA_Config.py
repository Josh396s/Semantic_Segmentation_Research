import loralib as lora

def lora_config(model, ranking):
    #VISION ENCODER
    #Patch embedding
    # model.vision_encoder.patch_embed.projection = lora.Conv2d(3, 768, kernel_size=16, stride=(16, 16), r = rank)

    for layer in model.vision_encoder.layers:
        #Attention block
        layer.attn.qkv = lora.MergedLinear(768, 2304, r=ranking, enable_lora=[True, True, True])
        # layer.attn.proj = lora.Linear(768, 768, r=rank)

        #MLP block
        # layer.mlp.lin1 = lora.Linear(768, 3072, r=rank)
        # layer.mlp.lin2 = lora.Linear(3072, 768, r=rank)

    #Neck Layer
    # model.vision_encoder.neck.conv1 = lora.Conv2d(768, 256, kernel_size=1, stride=(1, 1), r=rank)
    # model.vision_encoder.neck.conv2 = lora.Conv2d(256, 256, kernel_size=3, stride=(1, 1), padding=(1, 1), r=rank)


    # MASK DECODER -----------------------------------------------------------------------------------------------------------
    for layer in model.mask_decoder.transformer.layers:
        #Self attention block
        layer.self_attn.q_proj = lora.Linear(256, 256, r=ranking)
        layer.self_attn.k_proj = lora.Linear(256, 256, r=ranking)
        layer.self_attn.v_proj = lora.Linear(256, 256, r=ranking)
    #     layer.self_attn.out_proj = lora.Linear(256, 256, r=rank)

    #     #Cross attention block (Token -> Image)
    #     layer.cross_attn_token_to_image.q_proj = lora.Linear(256, 128, r=rank)
    #     layer.cross_attn_token_to_image.k_proj = lora.Linear(256, 128, r=rank)
    #     layer.cross_attn_token_to_image.v_proj = lora.Linear(256, 128, r=rank)
    #     layer.cross_attn_token_to_image.out_proj = lora.Linear(128, 256, r=rank)

    #     #MLP block
    #     layer.mlp.lin1 = lora.Linear(256, 2048, r=rank)
    #     layer.mlp.lin2 = lora.Linear(2048, 256, r=rank)

    #     #Cross attention block (Image -> Token)
    #     layer.cross_attn_image_to_token.q_proj = lora.Linear(256, 128, r=rank)
    #     layer.cross_attn_image_to_token.k_proj = lora.Linear(256, 128, r=rank)
    #     layer.cross_attn_image_to_token.v_proj = lora.Linear(256, 128, r=rank)
    #     layer.cross_attn_image_to_token.out_proj = lora.Linear(128, 256, r=rank)

    #Final attention block (Token -> Image)
    # model.mask_decoder.transformer.final_attn_token_to_image.q_proj = lora.Linear(256, 128, r=rank)
    # model.mask_decoder.transformer.final_attn_token_to_image.k_proj = lora.Linear(256, 128, r=rank)
    # model.mask_decoder.transformer.final_attn_token_to_image.v_proj = lora.Linear(256, 128, r=rank)
    # model.mask_decoder.transformer.final_attn_token_to_image.out_proj = lora.Linear(128, 256, r=rank)

    return model