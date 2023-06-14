import os

try:
    from engine.libs.hub import get_last_weight, get_best_weight
except ImportError:
    raise ImportError(
        'Please run "pip install granular-engine" to install engine')


def weight_and_experiment(url, best=False):
    if best:
        checkpoint, experiment_id = get_best_weight(url)
    else:
        checkpoint, experiment_id = get_last_weight(url)

    dst_path = 'pretrain/' + '/'.join(checkpoint.replace('gs://', '').replace('s3://', '').split('/')[2:])
    os.system(f"gsutil -m cp -n -r {checkpoint} {dst_path}")

    experiment_path = 'gs://' + '/'.join(checkpoint.replace('gs://', '').replace('s3://', '').split('/')[:-1])
    experiment_config_path = 'pretrain/' + '/'.join(checkpoint.replace('gs://', '').replace('s3://', '').split('/')[2:-1])
    os.system(f"gsutil -m cp -n -r {experiment_path}/epxeriment_config.json {experiment_config_path}/")
            
    return dst_path, experiment_id
