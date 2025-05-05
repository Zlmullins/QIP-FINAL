
'''
#################################################

Group 1 Shor's Algorithm Implementation
Josh, Serina, Xiao, Morgan, and Zoe
1. This program intends to acheive the following:
Create a Jupyter notebook with input a number to be factored and output the pair
of factors (or a failure message)
2. Show how the size of the quantum circuit scales as a function of the number of
bits required to represent the input number.
3. Show how the classical simulation breaks down due to the exponential scaling of
the size of the Hilbert space
4. If you have time, discuss different possible ways to implement the modular
exponentiation function on a quantum computer and how that might impact the
circuit scaling

#################################################
'''


import numpy as np
from math import gcd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
import random
import time
import matplotlib.pyplot as plt

# Function to create a controlled modular exponentiation circuit for a^power mod 15
def c_amod15(a, power):
    U = QuantumCircuit(4)
    
    # Swap operations define the modular exponentiation behavior
    # These swaps are hardcoded for specific values of 'a' and 'power' (mod 15)
    for _ in range(power % 4):
        if a == 2:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        elif a == 7:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        elif a == 8:
            U.swap(0, 3)
            U.swap(1, 2)
        elif a == 11:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        elif a == 13:
            U.swap(1, 2)
            U.swap(0, 1)
        elif a == 14:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
    
    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
     # Create a controlled version of the gate
    c_U = U.control() 
    return c_U

'''
Shor’s algorithm often uses the number 15 in practice for simplicity in demonstrations.
The actual value used for modular exponentiation (mod 15) can vary depending on the factor being searched.
Modulo 15 is chosen in the example because it allows an efficient demonstration of Shor's algorithm 
using relatively simple calculations while still illustrating the key concepts of QPE and periodicity.
The number 15 has convenient prime factors (3 and 5), making it easier to demonstrate the algorithm's power.
This idea was originated from IBM Qiskit documentation https://github.com/Qiskit/textbook/blob/main/notebooks/ch-algorithms/shor.ipynb
'''

# Function for Quantum Phase Estimation (QPE) on a^power mod 15
def qpe_amod15(a):
    n_count = 8  # The number of qubits used for counting
    qc = QuantumCircuit(n_count + 4, n_count)  # Create a quantum circuit with n_count qubits for counting and 4 for modular exponentiation

    # Apply Hadamard to the counting qubits to create superposition
    for q in range(n_count):
        qc.h(q)

    qc.x(n_count + 3)  # Initialize the auxiliary qubit in the state |1>

    # Apply the controlled modular exponentiation gates for each value of the counting qubits
    for q in range(n_count):
        qc.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])

    # Apply inverse Quantum Fourier Transform (QFT) to extract the phase
    qc.append(QFT(num_qubits=n_count, inverse=True), range(n_count))
    
    # Measure the output qubits to get the phase information
    qc.measure(range(n_count), range(n_count))

    # Simulate the quantum circuit
    sim = AerSimulator()
    qc = transpile(qc, sim)  
    result = sim.run(qc, shots=1).result() 
    counts = result.get_counts() 
    return list(counts.keys())[0]  

# Function to extract the order from the phase string output by QPE
def get_order(phase_str, N):
    # Convert the binary phase string to a decimal phase
    phase = int(phase_str, 2) / 2**len(phase_str)
    
    # If the phase is 0, the order is invalid (we want a non-zero phase)
    if phase == 0:
        return None
    
    # The order is the inverse of the phase value, rounded to the nearest integer
    frac = round(1 / phase)  
    return frac

# Main function to run Shor's algorithm for integer N
def shor(N):
    timeout = time.time() + 60  
    while time.time() < timeout:
        if N % 2 == 0:
            return 2, N // 2
        a = random.randint(2, N - 1) 
        
        # If gcd(a, N) is greater than 1, then we have found a factor
        if gcd(a, N) > 1:
            return gcd(a, N), N // gcd(a, N)
        
        # Perform quantum phase estimation to find the period of a modulo N
        phase = qpe_amod15(a)
        r = get_order(phase, N)  
        
        # If the order is invalid (None) or the order is not even, continue
        if r is None or r % 2 != 0:
            continue
        
        # Compute possible factors using the order 'r'
        guess1 = gcd(pow(a, r // 2) - 1, N)
        guess2 = gcd(pow(a, r // 2) + 1, N)

        # If either guess is trivial (1 or N), skip this iteration
        if guess1 in [1, N] or guess2 in [1, N]:
            continue
        else:
            # If both guesses are valid, return them as the factors
            return guess1, guess2
    return None  # Return None if factoring fails

# Function to create a modular exponentiation quantum circuit for a^power mod N
def controlled_modular_exponentiation(a, N, power):
    """Creates a modular exponentiation circuit for a^power mod N."""
    n = N.bit_length()
    # Number of qubits needed to represent N
    qc = QuantumCircuit(n) 
    
    # Apply placeholder operations (controlled NOT gates) for modular exponentiation
    # These would be replaced with actual operations for a real implementation
    for _ in range(power):
        for i in range(n):
            # Placeholder operation 
            qc.cx(i, (i+1)%n)  
    
    U = qc.to_gate()  
    U.name = f"{a}^{power} mod {N}"  
    return U.control() 

# Function to generate the QPE circuit for modular exponentiation
def qpe_for_modular_exponentiation(a, N):
    n_count = 8  # Number of counting qubits
    n = N.bit_length()  # Number of qubits for N (typically log2(N))

    # Create quantum circuit with n_count counting qubits and n qubits for the modular exponentiation
    qc = QuantumCircuit(n_count + n, n_count)

    # Apply Hadamard to the counting qubits
    qc.h(range(n_count))

    # Initialize the bottom register to |1>
    qc.x(n_count + n - 1)

    # Apply controlled-U^2^j operations for each counting qubit
    for i in range(n_count):
        power = 2 ** i
        cmod = controlled_modular_exponentiation(a, N, power)  
        qc.append(cmod, [i] + list(range(n_count, n_count + n)))

    # Apply inverse Quantum Fourier Transform
    qc.append(QFT(n_count, inverse=True), range(n_count))
    # Measure the output qubits
    qc.measure(range(n_count), range(n_count))  
    return qc

# Function to visualize the Shor's algorithm quantum circuit
def visualize_shors_circuit(N, a):
    """Generates and visualizes the Shor's Algorithm quantum circuit for factoring N."""
    # Generate the QPE circuit for modular exponentiation
    qc = qpe_for_modular_exponentiation(a, N)

    # Visualize the quantum circuit using matplotlib
    qc.draw('mpl')  # Use 'mpl' to render as a matplotlib figure
    plt.title(f"Quantum Circuit for Shor's Algorithm (Factoring Given Value)")
    plt.show()
    # Display the circuit ad save the image
    plt.savefig('Shor.png')  
    
# Main function to get input and run Shor's algorithm
if __name__ == "__main__":
    try:
        N = int(input("Enter an integer number N to factor: ")) 
        if N <= 1:
            raise ValueError("N must be greater than 1.")  
        # Run Shor's algorithm to factor N
        factors = shor(N)  
        print(f"Factors of {N}:", factors if factors else "Failed to factor")
    
    #Handle errors
    except Exception as e:
        print("Error:", e)
    # Choose a co-prime number a (must be coprime to N)
    a = 2
    #Print shor
    visualize_shors_circuit(N, a)  

