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
    os.system(f"gsutil -m cp -n -r {checkpoint} {dst_path} 2> /dev/null")
            
    return dst_path, experiment_id
