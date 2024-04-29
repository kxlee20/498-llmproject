config_args = {
    'task': "commonsense",
    'data_dir': './dataset',
    'train_dataset': None,
    'eval_dataset': None,
    'model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'seed': 42,
    'layers': '4',
    'rank': 1,
    'position': 'f1+l1',
    'epochs': 1,
    'is_wandb': False,
    'wandb_name': 'reft',
    'save_model': False,
    'max_n_train_example': 50,
    'max_n_eval_example': 10,
    'type': 'LoreftIntervention',
    'gradient_accumulation_steps': 4,
    'batch_size': 4,
    'eval_batch_size': 4,
    'output_dir': './official_results',
    'lr': 0.005,
    'schedule': 'linear',
    'wu': 0.00,
    'wd': 0.00,
    'dropout': 0.00,
    'act_fn': None,
    'add_bias': False,
    'test_split': 'test',
    'train_on_inputs': False,
    'max_length': 512,
    'use_normalized_template': False,
    'allow_cls_grad': False,
    'metric_for_best_model': 'accuracy',
    'dtype': 'bfloat16',  # Assumes `torch` has been imported
    'logging_steps': 1,
    'wandb_dir': 'wandb',
    'wandb_proj': 'MyReFT',
    'share_weights': False,
    'greedy_decoding': False,
    'temperature': None,
    'top_p': None,
    'top_k': None,
    'commonsense': {
        'train_datasets': [
            'piqa_train'
        ],
        'eval_datasets': [
            'piqa'
        ],
        'task_prompt_template': '%s\n',
        'trigger_tokens': 'the correct answer is ',
        'generation_args': {
            # align with https://github.com/AGI-Edgerunners/LLM-Adapters
            True: {
                'max_new_tokens': 32,
                'do_sample': False,
            },
            False: {
                'max_new_tokens': 32,
                'temperature': 0.1,
                'top_p': 0.75,
                'top_k': 40,
                'num_beams': 4,
                'do_sample': True,
            }
        }
    }

}
