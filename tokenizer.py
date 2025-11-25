from lightning.pytorch.cli import LightningCLI
import sys

from genie import VideoTokenizer
from genie.dataset import LightningPlatformer2D

def cli_main():
    '''
    Main function for the training script.
    '''
    # Map 'train' command to 'fit' for convenience
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv[1] = 'fit'
    
    # That's all it takes for LightningCLI to work!
    # No need to call .fit() or .test() or anything like that.
    cli = LightningCLI(
        VideoTokenizer,
        LightningPlatformer2D,
    )

if __name__ == '__main__':    
    cli_main()