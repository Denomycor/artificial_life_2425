import torch


def init_torch():
    torch.manual_seed(40)
    if(torch.cuda.is_available()):
        torch.set_default_device("cuda")
        print("Enabled cuda as default device")


# TODO: Entry point
def main():
    init_torch()


if __name__ == "__main__":
    main()

