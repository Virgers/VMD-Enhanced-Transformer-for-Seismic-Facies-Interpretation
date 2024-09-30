from collections import Counter

best_setup_indices = []
def model_selection(mse, mae, training_times):
    # length of the whole sets
    # Benchmark setup (setup 0)
    benchmark_mse = mse[0]
    benchmark_mae = mae[0]

    # Calculate relative improvements
    relative_improvements = [(round(benchmark_mse - mse[i], 5), round(benchmark_mae - mae[i],5)) for i in range(len(mse))]
    print("Relative improvements (MSE, MAE):", relative_improvements)

    # Normalize improvements (optional)
    max_mse_improvement = max([improvement[0] for improvement in relative_improvements])
    max_mae_improvement = max([improvement[1] for improvement in relative_improvements])

    normalized_improvements = [(round(improvement[0] / max_mse_improvement,5), round(improvement[1] / max_mae_improvement ,5))
                            for improvement in relative_improvements]
    print("Normalized improvements (MSE, MAE):", normalized_improvements)

    weights = (0.5, 0.5)  # Equal weight for MSE and MAE
    # Calculate combined improvement metric
    combined_improvements = [weights[0] * imp[0] + weights[1] * imp[1] for imp in normalized_improvements]

    # Calculate cost-benefit ratio
    cost_benefit_ratios = [imp / (time) for imp, time in zip(combined_improvements, training_times)]

    # Select the setup with the highest cost-benefit ratio
    best_setup_index = cost_benefit_ratios.index(max(cost_benefit_ratios))
    # best index here need to add 1 to coincide with the training_times
    print(f"Best setup index: {best_setup_index}")
    print(f"Training time: {training_times[best_setup_index]:.5f}")
    # print(f"FLOPs: {flops[best_setup_index]:.5e}")
    print(f"Cost-benefit ratio: {cost_benefit_ratios[best_setup_index]:.5e}")

    best_setup_index = cost_benefit_ratios.index(max(cost_benefit_ratios))
    # Add the best setup index to the list (adjust index to be 1-based if needed)
    best_setup_indices.append(best_setup_index) 
    return best_setup_indices


def data_lib(model, mask_rate):
    if model == 'ITransformer':
    # iTransformer 0.125
        if mask_rate == 0.125:
            training_times = [16.67, 22.047, 27.427, 33.189, 38.785, 44.321, 49.749]  # in seconds
            mse = [0.54613, 0.44347, 0.37089, 0.35145, 0.33098, 0.28573, 0.28518]  # Mean Squared Error
            mae = [0.54247, 0.49071, 0.45060, 0.43521, 0.42206, 0.39290 , 0.39217 ]
        if mask_rate == 0.25:
        # iTransformer 0.25
            training_times = [16.615, 22.117, 27.543, 33.188, 38.834, 44.418, 49.879]
            mse = [0.46684, 0.40533, 0.36929, 0.33923, 0.34209, 0.32214, 0.32978]
            mae = [0.50111 , 0.46798 , 0.44521 , 0.42565 , 0.42820 , 0.41454 , 0.41835 ]

        # iTransformer 0.375
        if mask_rate == 0.375:
            training_times = [16.671, 22.250, 27.772, 33.139, 38.795, 44.300, 49.975]
            mse = [0.51776, 0.45278, 0.42186, 0.40351, 0.39160, 0.38921, 0.38635]
            mae = [0.52432 , 0.49251 , 0.47133 , 0.45981 , 0.45377 , 0.45197 , 0.44953 ]

        # iTransformer 0.5
        if mask_rate == 0.5:
            training_times = [16.801, 22.149, 27.427, 33.219 , 38.964, 44.272, 49.870]
            mse = [0.59990, 0.54427, 0.48951, 0.47129, 0.45535, 0.45481, 0.45212]
            mae = [0.56121 , 0.53537 , 0.50468 , 0.49673 , 0.48749 , 0.48698 , 0.48517 ]

    if model == 'NST':

        if mask_rate == 0.125:
            #NST 0.125
            training_times = [22.127, 29.563, 35.871, 43.394, 52.774, 57.337, 64.647]
            mse = [0.85186, 0.81909, 0.78678, 0.75089, 0.72746, 0.70647, 0.68461]
            mae = [0.65630 , 0.64565 , 0.63263 , 0.61435 , 0.60596 , 0.59940 , 0.59008 ]

        if mask_rate == 0.25:
            training_times = [21.539, 28.793, 35.948, 45.309, 52.619, 57.278, 64.647]
            mse = [0.86608, 0.83284, 0.79971, 0.76808, 0.74411, 0.68633, 0.68461]
            mae = [0.65771 , 0.64534 , 0.63286 , 0.62155 , 0.61145 , 0.59330 , 0.59008 ]

        if mask_rate == 0.375:
            # NST 0.375
            training_times = [21.812, 28.784, 35.844, 45.374, 52.598, 57.309, 64.582]
            mse = [0.87838, 0.83027, 0.81356, 0.78302, 0.78028, 0.75130, 0.73162]
            mae = [0.66402 , 0.64277 , 0.63860 , 0.62778 , 0.62399 , 0.61314 , 0.60648 ]

        if mask_rate == 0.5:  # NST 0.5
            training_times = [21.592, 29.705, 35.956, 45.321, 50.465, 57.453, 64.423]
            mse = [0.91257, 0.88736, 0.85598, 0.84545, 0.79633, 0.80447, 0.77586]
            mae = [0.67568 , 0.66727 , 0.65541, 0.65032 , 0.63240 , 0.63575 , 0.62689 ]

    # elif model =='autoformer':
    #     # autoformer
    #     if mask_rate == 0.125:
    #         training_times = [11.836 , 18.553, 24.381, 29.581, 33.357, 36.609, 42.851]
    #         mse = [1.18838, 1.16657, 1.16283, 1.19360, 1.23356, 1.20099, 1.16533]
    #         mae = [0.77011 , 0.75782 , 0.75447 , 0.76929  , 0.78534 , 0.77178 , 0.76213 ]
    #     if mask_rate == 0.25:
    #         training_times = [14.496, 18.060, 23.731, 28.109, 33.964, 38.254, 42.831]
    #         mse = [1.21854, 1.21472, 1.20690, 1.15592, 1.13379, 1.13660, 1.24881]
    #         mae = [0.78243 , 0.77945 , 0.77560 , 0.76388 , 0.75375 , 0.75494 , 0.80345 ]
    #     if mask_rate == 0.375:
    #         training_times = [15.123, 23.749, 28.864, 34.041,37.736,44.515]
    #         mse = [1.20061, 1.18569, 1.20107, 1.19601, 1.16219, 1.20954, 1.25679]
    #         mae = [0.78354 , 0.77715 , 0.78274 , 0.78091 , 0.76788 , 0.78549 , 0.80713 ]
    #     if mask_rate == 0.5:
    #         training_times = [13.553,17.762,23.348,28.686,33.288,37.655,44.445]
    #         mse = [1.21654, 1.23105, 1.24673, 1.20104, 1.24606, 1.25322, 1.32919]
    #         mae = [0.78959 , 0.79312 , 0.79489 , 0.78331 , 0.80064 , 0.81987 , 0.82528 ]

    elif model =='FED':
        if mask_rate == 0.125: 
        # FED former
            training_times = [72.237, 94.542, 115.740 , 138.798, 165.193, 182.664, 203.324]
            mse = [0.71550, 0.69407, 0.68619, 0.67429, 0.66340, 0.66026, 0.64484]
            mae = [0.63814 , 0.62818 , 0.62535 , 0.61986 , 0.61540 , 0.61381 , 0.60644 ]
        if mask_rate == 0.25:
            training_times = [ 71.954, 95.763, 114.595, 143.463, 168.785, 181.979, 203.985]
            mse = [0.92160, 0.90017, 0.89801, 0.88228, 0.87675, 0.86774, 0.85492]
            mae = [0.71506 , 0.70692 , 0.70584 , 0.70068 , 0.69892 , 0.69425 , 0.69035 ]
        if mask_rate == 0.375:
            training_times = [ 71.383, 92.494, 110.270, 142.130, 167.700, 186.710, 209.726 ]
            mse = [1.00574, 0.99042, 0.98384, 0.97454, 0.96251, 0.95400, 0.94709]
            mae = [0.74281 , 0.73861 , 0.73570 , 0.73182 , 0.72787 , 0.72452 , 0.72170 ]
        if mask_rate == 0.5:
            training_times = [ 70.136, 101.554, 121.934 , 139.553, 167.218, 185.121, 210.875 ]
            mse = [1.05644, 1.04275, 1.02805, 1.02924, 1.02244, 1.01576, 1.00752]
            mae = [0.75937, 0.75490, 0.74852 , 0.74964 , 0.74673 , 0.74397 , 0.74057 ]

    elif model =='informer':
        if mask_rate == 0.125: 
        # FED former
            training_times = [170.480 ,  257.737, 344.133 , 379.937, 393.368, 436.716, 466.643]
            mse = [1.00203, 0.93478, 0.95278,  0.90397, 0.91562, 0.90129, 0.86069]
            mae = [0.71241  , 0.69703  , 0.70546  , 0.68531 , 0.69136 ,  0.68239 , 0.67056 ]
        if mask_rate == 0.25:
            training_times = [ 194.114, 275.494, 365.208,  383.569, 419.066, 447.585, 472.452]
            mse = [1.00664, 0.97599, 0.89756, 0.93593,  0.91562, 0.90773, 0.87659]
            mae = [0.71801 , 0.71174 , 0.68034 ,  0.69350 , 0.69136 ,  0.68586 , 0.67398 ]
        if mask_rate == 0.375:
            training_times = [ 159.462, 262.335, 364.201, 393.605, 423.865, 558.044, 562.324 ]
            mse = [1.03075, 0.97103, 0.95758, 0.93494, 0.92906, 0.89525, 0.92633]
            mae = [ 0.73094 ,  0.70785 ,  0.70193 ,  0.69185,  0.69158 , 0.68001 , 0.69154 ]
        if mask_rate == 0.5:
            training_times = [199.429, 225.599, 313.328,372.762, 414.772, 396.722, 610.518]
            mse = [1.01513, 0.98909, 0.98296, 0.95501, 0.95673, 0.93734, 0.91115]
            mae = [ 0.72500 ,  0.71415 ,  0.70945 ,  0.70133 ,  0.70676, 0.69492, 0.68291  ]

    return mse, mae, training_times


def main(model, mask_rate):
    mse, mae, training_times = data_lib(model, mask_rate)
    best_setup_indices = model_selection(mse, mae, training_times)
    
    counter = Counter(best_setup_indices)
    most_common_indices = counter.most_common(3)

    for index, count in most_common_indices:
        print(f"Index: {index}, Count: {count}")


if __name__ =='__main__':
    models = ['NST', 'FED', 'ITransformer', 'informer']
    # List of mask rates
    mask_rates = [0.125, 0.25, 0.375, 0.5]

    # Iterate through each model and each mask rate
    for model in models:
        for mask_rate in mask_rates:
            print(f"Processing model: {model}, mask rate: {mask_rate}")
            main(model, mask_rate)
