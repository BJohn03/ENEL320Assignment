import numpy as np
import matplotlib.pyplot as plt
import time

'''This assumes and indices starting at zero in position 0
'''
def compute_dft(signal):
    
    length = len(signal)
    indices = np.arange(length)
    result = np.array([])

    for k in indices:
        result_val = 0
        n = 0
        for value in signal:
            result_val += value * np.exp((-2j*np.pi*k*n) / length)
            n += 1
        result = np.append(result, [result_val])
    
    return result

def main():
    
    input_sig = np.array([2,3,1,4,3,4,4,4,3,1,3,4,2,1,1])
    result = np.round(compute_dft(input_sig), 3)
    print(result)
    
def test_my_dft_func():
    
    arrays = np.random.randint(0, 100, (1000,1000))
    
    j = 0
    for array in arrays:
        print(f"{j}% Complete")
        my_result = compute_dft(array)
        real_result = np.fft.fft(array)
        
        my_result = np.round(my_result, 3)
        real_result = np.round(real_result, 3)
        
        for i in range(0, 100):
            if my_result[i] != real_result[i]:
                raise ValueError(f"Value {my_result[i]} does not equal {real_result[i]} at index {i}.")
        j += 1

def performance_benchmark():
    
    start = 1
    stop_fast = 1000000
    stop_slow = 1000
    step_fast = 100000
    step_slow = 100
    
    results_slow = np.array([])
    results_fast = np.array([])
    
    sizes_fast = np.arange(start, stop_fast, step_fast)
    sizes_slow = np.arange(start, stop_slow, step_slow)
    
    for i in sizes_fast:
        
        print(f"Fast {i/stop_fast * 100}% Complete")
        arrays = np.random.randint(0, 20, (100,i))

        before = time.time()
        for array in arrays:
            np.fft.fft(array)
        after = time.time()
        
        difference = after - before
        results_fast = np.append(results_fast, difference)
        
        print(f"Fast DFT Function completed in {after-before} seconds")
    
    for i in sizes_slow:
        
        print(f"Slow {i/stop_slow * 100}% Complete")
        arrays = np.random.randint(0, 20, (100,i))
        
        before = time.time()
        for array in arrays:
            compute_dft(array)
        after = time.time()
        
        difference = after - before
        results_slow = np.append(results_slow, difference)
        
        print(f"Slow DFT Function completed in {difference} seconds")
    
    fig, ax = plt.subplots()
    ax.set_xlabel("100 arrays of size N")
    ax.set_ylabel("Time to compute (s)")
    ax.plot(sizes_slow, results_slow, "bo", label="Slow DFT Function")
    ax.plot(sizes_fast, results_fast, "ro", label="Fast DFT Function")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
    performance_benchmark()