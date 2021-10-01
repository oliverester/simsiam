import os
import simsiam.builder
import torch 
import torchvision.models as models

def load_pretrained_model(args):
    
    args.device = 'cuda'
    torch.cuda.set_device(args.gpu)

    print("=> creating model '{}'".format(args.arch))
    
    # get model but without predictor (we want the features)
    model = simsiam.builder.SimSiam(models.__dict__[args.arch],
                                    args.dim,
                                    inference=True)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder'): # and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                else:
                    print(f"Remove layer {k}")
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == set([])

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model.cuda(args.gpu)
    
    return model
