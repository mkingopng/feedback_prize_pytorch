"""

"""

from scratch_config import *
from scratch_functions import *


if __name__ == "__main__":
    # loop to train model (or load model)
    if not LOAD_MODEL_FROM:
        for epoch in range(config['epochs']):
            print(f"### Training epoch: {epoch + 1}")
            for g in optimizer.param_groups:
                g['lr'] = config['learning_rates'][epoch]
            lr = optimizer.param_groups[0]['lr']
            print(f'### LR = {lr}\n')
            train(epoch)
            torch.cuda.empty_cache()
            gc.collect()
        torch.save(model.state_dict(), f'bigbird_v{VER}.pt')
    else:
        model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/bigbird_v{VER}.pt'))
