# Алгоритм Виттерби
import numpy as np

class MLSE_BPSK:
    """
    MLSE эквалайзер для BPSK с памятью канала 1
    Максимально просто и понятно
    """
    
    def __init__(self, h0, h1):
        self.h0 = h0
        self.h1 = h1
        
    def equalize(self, received):
        
        metric_plus = 0      # ошибка для состояния "+1"
        metric_minus = 0     # ошибка для состояния "-1"
        path_plus = []       # история для состояния "+1"
        path_minus = []      # история для состояния "-1"
        
        for r in received:
            print(f"\n--- Принят символ {r:.2f} ---")
            
            new_metric_plus = float('inf')
            new_metric_minus = float('inf')
            new_path_plus = []
            new_path_minus = []
            
            # 1. Было состояние +1, отправляем +1
            #    ожидаемый сигнал = h0*(+1) + h1*(+1) = h0 + h1
            expected = self.h0*1 + self.h1*1
            error = (r - expected)**2
            candidate = metric_plus + error
            print(f"  Из [+1] с [+1] → новое [+1]: ошибка={error:.3f}, всего={candidate:.3f}")
            if candidate < new_metric_plus:
                new_metric_plus = candidate
                new_path_plus = path_plus + [1]
            
            # 2. Было состояние -1, отправляем +1
            #    ожидаемый сигнал = h0*(+1) + h1*(-1) = h0 - h1
            expected = self.h0*1 + self.h1*(-1)
            error = (r - expected)**2
            candidate = metric_minus + error
            print(f"  Из [-1] с [+1] → новое [+1]: ошибка={error:.3f}, всего={candidate:.3f}")
            if candidate < new_metric_plus:
                new_metric_plus = candidate
                new_path_plus = path_minus + [1]
            
            # 3. Было состояние +1, отправляем -1
            #    ожидаемый сигнал = h0*(-1) + h1*(+1) = -h0 + h1
            expected = self.h0*(-1) + self.h1*1
            error = (r - expected)**2
            candidate = metric_plus + error
            print(f"  Из [+1] с [-1] → новое [-1]: ошибка={error:.3f}, всего={candidate:.3f}")
            if candidate < new_metric_minus:
                new_metric_minus = candidate
                new_path_minus = path_plus + [-1]
            
            expected = self.h0*(-1) + self.h1*(-1)
            error = (r - expected)**2
            candidate = metric_minus + error
            print(f"  Из [-1] с [-1] → новое [-1]: ошибка={error:.3f}, всего={candidate:.3f}")
            if candidate < new_metric_minus:
                new_metric_minus = candidate
                new_path_minus = path_minus + [-1]
            
            # Обновляем метрики и пути
            metric_plus = new_metric_plus
            metric_minus = new_metric_minus
            path_plus = new_path_plus
            path_minus = new_path_minus
            
            print(f"  Лучший путь в [+1]: ошибка={metric_plus:.3f}, история={path_plus}")
            print(f"  Лучший путь в [-1]: ошибка={metric_minus:.3f}, история={path_minus}")
        
        print(f"  Состояние [+1]: ошибка={metric_plus:.3f}, путь={path_plus}")
        print(f"  Состояние [-1]: ошибка={metric_minus:.3f}, путь={path_minus}")
        
        if metric_plus < metric_minus:
            print(f"  Победило состояние [+1]")
            return path_plus
        else:
            print(f"  Победило состояние [-1]")
            return path_minus


h0 = 0.8
h1 = 0.5

tx = [1, -1, 1, -1, 1]

rx = []
rx.append(h0 * tx[0])
for k in range(1, len(tx)):
    rx.append(h0 * tx[k] + h1 * tx[k-1])

print("Сигнал после канала (без шума):", [f"{x:.2f}" for x in rx])

# Добавляем немного шума для реализма
np.random.seed(42)
noise = 0.8 * np.random.randn(len(rx))
rx_noisy = [rx[i] + noise[i] for i in range(len(rx))]

print("Сигнал с шумом:          ", [f"{x:.2f}" for x in rx_noisy])
print()

eq = MLSE_BPSK(h0, h1)
decoded = eq.equalize(rx_noisy)

print("  Передано:    ", tx)
print("  Восстановлено:", decoded)
print("  Совпадает?   ", tx == decoded)