import numpy as np
# import matplotlib.pyplot as plt

def predict_best_lr(lr_data, loss_data):
    
    # 2. Transform the learning rates into log10 space
    log_lr = np.log10(lr_data)

    # 3. Fit a 2nd-degree polynomial (parabola): y = a*(log_x)^2 + b*(log_x) + c
    a, b, c = np.polyfit(log_lr, loss_data, 2)

    # 4. Calculate the vertex (minimum point of the parabola)
    # Formula for axis of symmetry: x = -b / (2a)
    log_lr_opt = -b / (2 * a)
    lr_opt = 10**log_lr_opt
    predicted_min_loss = a * (log_lr_opt**2) + b * log_lr_opt + c

    print("=== POWER LAW SWEEP CALCULATIONS ===")
    print(f"Predicted Optimal Peak LR: {lr_opt:.5f}")
    print(f"Predicted Minimum Loss:    {predicted_min_loss:.5f}")

# 1. Define your sweep data points (peak_lr, best_loss)
lr_data = np.array([0.0001, 0.0017, 0.0030])
loss_data = np.array([2.3965, 1.50816, 1.46965])

def visualize_best_lr(a, b, c):
    # 5. Generate smooth curve for visualization
    smooth_log_lrs = np.linspace(np.log10(0.00005), np.log10(0.01), 200)
    predicted_losses = a * (smooth_log_lrs**2) + b * smooth_log_lrs + c

    # Plot the results
    # plt.figure(figsize=(8, 5))
    # plt.scatter(lr_data, loss_data, color='red', s=100, label='Your Sweep Runs', zorder=5)
    # plt.plot(10**smooth_log_lrs, predicted_losses, color='blue', linestyle='--', label='Fitted Power Law Curve')
    # plt.axvline(lr_opt, color='green', linestyle=':', label=f'Optimal LR ({lr_opt:.5f})')

    # plt.xscale('log')
    # plt.xlabel('Peak Learning Rate (Log Scale)')
    # plt.ylabel('Validation Loss')
    # plt.title('LLM Learning Rate Optimization Curve')
    # plt.legend()
    # plt.grid(True, which="both", ls="-", alpha=0.2)
    # plt.show()

def main(full_lr_data, full_loss_data):

    nn = len(full_loss_data)
    assert nn == len(full_loss_data)
    for start in range(nn - 2):
        stop = start + 3
        lr_window = full_lr_data[start:stop]
        loss_window = full_loss_data[start:stop]
        predict_best_lr(lr_window, loss_window)
        print('---------------')


peak_lrs = [
    0.0001, 0.0002, 0.0003, 0.0005, 0.0008,
    0.0010, 0.0013, 0.0017, 0.0022, 0.0030, 0.0040, 0.0050
]

best_validation_losses = [
    2.39650235, 2.09355928, 1.91394936, 1.73347608,
    1.61671751, 1.57586616, 1.53634123, 1.50816332,
    1.49082888, 1.46965221, 1.45742784, 1.45390940
]
if __name__ == '__main__':
    main(peak_lrs, best_validation_losses)
