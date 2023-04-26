brooks_corey = BrooksCorey()
w_train_data = CapPressDataset(len=200)
w_train_dataloader = DataLoader(w_train_data, batch_size=200)
x = torch.arange(0, 1, 0.01).unsqueeze(-1)
truth = brooks_corey(x)
plt.plot(x.numpy(force=True), truth.numpy(force=True), label="Ground Truth")
x, y = next(iter(w_train_dataloader))
plt.scatter(
    x.numpy(force=True),
    y.numpy(force=True),
    label="Data",
)
plt.legend()
plt.savefig(os.path.join("saved_models", "BaseNN_CapPressHysteresis_data.png"))
