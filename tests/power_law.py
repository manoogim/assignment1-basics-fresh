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


peak_lr_arr = [
    0.0001,
    0.0002,
    0.0003,
    0.0005,
    0.0008,
    0.0010,
    0.0013,
    0.0017,
    0.0022,
    0.0030,
    0.0040,
    0.0050,
    0.0060,
    0.0070,
]

best_loss_arr = [
    2.3965023517608643,
    2.0935592770576480,
    1.9139493584632872,
    1.7334760785102843,
    1.6167175054550171,
    1.5758661627769470,
    1.5363412261009215,
    1.5081633210182190,
    1.4908288836479189,
    1.4696522116661073,
    1.4574278354644776,
    1.4539093971252440,
    1.4563342809677124,
    1.4650454521179200,
]
if __name__ == '__main__':
    main(peak_lr_arr, best_loss_arr)
