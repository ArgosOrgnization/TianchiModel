# TianchiModel

A repo to learn tianchi competition's model.

## Problems you may meet

### Pretrained-Model problem

Pretrained model is usually downloaded from the internet by a link. However, as the network is not stable, downloading the model may fail. A broken model may cause the program to crash. To solve this problem, you can delete the broken model and download it again.

On linux, the broken model is usually stored in `~/.cache/torch/hub/checkpoints/`. Just delete the broken model and run the script again.