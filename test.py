import datetime

def create_experiment_settings(args):
    # Format the date and time
    formatted_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the settings dictionary
    settings = {
        "datetime": formatted_date_time,
        "model": args.model,
        "dataset": args.dataset,
        "train_prop": f"{args.train_proportion:.2f}",
        "test_prop": f"{args.test_proportion:.2f}",
        "mask_rate": f"{args.mask_rate:.2f}",
        "is_vmd": str(args.is_vmd),
        "embed": args.embedding_flag,
        "epochs": str(args.train_epochs),
        "batch": str(args.batch_size)
    }
    
    # Create a directory-friendly string
    dir_name = "_".join([f"{k}-{v}" for k, v in settings.items()])
    
    return settings, dir_name

# Example usage:
class Args:
    def __init__(self):
        self.model = "transformer"
        self.dataset = "cifar10"
        self.train_proportion = 0.8
        self.test_proportion = 0.2
        self.mask_rate = 0.15
        self.is_vmd = True
        self.embedding_flag = "concat"
        self.train_epochs = 100
        self.batch_size = 32

args = Args()
settings, dir_name = create_experiment_settings(args)

print("Settings dictionary:", settings)
print("Directory name:", dir_name)