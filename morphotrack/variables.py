settings = {
    'pyramid_levels':3,
    'pyramid_stop_level': 0,
    'step_size': [1.0, 1.0, 1.0],
    'block_size': [512,512,512],
    'block_energy_epsilon':1e-7,
    'max_iteration_count':100,
    'constraints_weight':1000.0,
    'regularization_weight': 0.25, # default 0.25
    'regularization_scale': 1.0, # default 1.0
    'regularization_exponent': 2.0, # default 2.0
    'image_slots':[{
            'resampler': 'gaussian',
            'normalize': True,
            'cost_function':[
                {
                    'function':'ncc',
                    'weight':1.0,
                    'radius':21
                }
            ]
        }]
}
