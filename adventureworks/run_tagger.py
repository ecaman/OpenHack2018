import click
from tagger.workflow.workflow_predict import workflow_predicting
from tagger.workflow.workflow_train import workflow_training
import glob

@click.command()
# Mandatory:
@click.option('--pipeline', type=click.Choice(['predicting', 'training']),
              help='Pipeline choice, either predicting or training')
@click.option('--input-path', help='Input file path')
@click.option('--output-path', help='For predicting, output file path')
# Optional
@click.option('--path-saves', default='./data/', help='Model path, for training it will be saved at this point,'
                                                      'For predicting it will load the most recent model at this point')

def main(pipeline, input_path, output_path, path_saves):
    """Program that will execute selected workflow for tagger"""


    if pipeline == 'predicting':
        preds = workflow_predicting(path_to_images=input_path, 
                                    path_to_saves=path_saves)
        
    elif pipeline == 'training':
        workflow_training(path_to_images=input_path, 
                          path_to_saves=path_saves)
        


if __name__ == '__main__':
    main()