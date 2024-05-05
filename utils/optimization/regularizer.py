class Regularizer:
    def __init__(self, lambda_depth, lambda_tot_widths, max_depth, max_width):
        self.lambda_depth = lambda_depth
        self.lambda_tot_widths = lambda_tot_widths
        self.max_depth = max_depth
        self.max_total_neurons = max_depth * max_width

    def __call__(self, score, network_architecture):
        complexity_depth_factor = len(network_architecture) - 2
        complexity_width_factor = sum(network_architecture[1:-1])

        normalized_complexity_depth_factor = (complexity_depth_factor / self.max_depth)
        normalized_complexity_width_factor = (complexity_width_factor / self.max_total_neurons)

        return score - (self.lambda_depth * normalized_complexity_depth_factor) - (self.lambda_tot_widths * normalized_complexity_width_factor)


MODEL_ARCHITECTURES_WEEDMAPPING = {
    'MiT-B0': [[32, 64, 160, 256], [2, 2, 2, 2]],  # [embed_dims, depths]
    'MiT-B1': [[64, 128, 320, 512], [2, 2, 2, 2]],

    'MiT-LD': [[16, 32, 80, 128], [2, 2, 2, 2]],  # Lightweight deep
    'MiT-L0': [[16, 32, 80, 128], [1, 1, 1, 1]],
    'MiT-L1': [[8, 16, 40, 64], [1, 1, 1, 1]],
    'MiT-L2': [[4, 8, 20, 32], [1, 1, 1, 1]],
}


class Regularizer_WeedMapping:
    def __init__(self, lambda_depths, lambda_widths, max_sum_depths=8, max_sum_widths=1024, model_architectures=MODEL_ARCHITECTURES_WEEDMAPPING):
        self.lambda_depths = lambda_depths
        self.lambda_widths = lambda_widths
        self.max_sum_depths = max_sum_depths
        self.max_sum_widths = max_sum_widths
        self.model_architectures = model_architectures

    def __call__(self, score, backbone_str):
        complexity_depths_factor = sum(self.model_architectures[backbone_str][1])
        complexity_widths_factor = sum(self.model_architectures[backbone_str][0])

        normalized_complexity_depths_factor = (complexity_depths_factor / self.max_sum_depths)
        normalized_complexity_widths_factor = (complexity_widths_factor / self.max_sum_widths)

        return score - (self.lambda_depths * normalized_complexity_depths_factor) - (self.lambda_widths * normalized_complexity_widths_factor)
