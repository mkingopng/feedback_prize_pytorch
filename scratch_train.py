from scratch_functions import *
import wandb


if __name__ == "__main__":
    # loop to train model (or load model)
    if not Params.OUTPUT_DIR:
        for epoch in range(Config.config['epochs']):
            print(f"### Training epoch: {epoch + 1}")
            for g in optimizer.param_groups:
                g['lr'] = Config.config['learning_rates'][epoch]
            lr = optimizer.param_groups[0]['lr']
            print(f'### LR = {lr}\n')
            train(epoch)
            torch.cuda.empty_cache()
            gc.collect()
        torch.save(model.state_dict(), f'longformer_v{Config.VER}.pt')
    else:
        model.load_state_dict(torch.load(f'{Params.OUTPUT_DIR}/longformer_v{Config.VER}.pt'))
        print('Model loaded.')
