# Welcome to Neural Networks in Elixir  

This project was created to deepen my understanding of Neural Networks and Elixir using the **Nx** library. I followed the amazing tutorial by [Aadi Malik](https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc).  

## 🚀 Overview  

This project explores how neural networks can be implemented in **Elixir** using Nx, drawing comparisons with Python-based implementations. The goal was to implement a **fully connected neural network** and overcome challenges in backpropagation due to Elixir’s functional nature.  

## 📌 Commands  

To run the project, use the following commands:  

```sh
elixirc neural_network.exs  # Runs the Elixir project
python3 example.py          # Runs the Python project
```

## 📚 Project Layout  

```plaintext
mkdocs.yml    # The configuration file.
docs/
    index.md  # The documentation homepage.
    ...       # Other markdown pages, images, and other files.
neural_network.exs
```

---

## 🏰️ Elixir: A Functional Language for Neural Networks  

Elixir is a **dynamic, functional programming language** designed for scalability and maintainability. However, working with Elixir for neural networks posed some unique challenges:  

### 🔍 Challenge: Everything in Elixir is an Expression  
Unlike Python, where **statements** exist, **Elixir treats everything as an expression**. This caused issues in backpropagation because my direct translation of Python code **did not update the weights correctly**.  

#### ❌ What went wrong?  
- My neural network wasn't updating the weights properly.
- The same weights were being analyzed repeatedly without modification.

### 📽️ Video Demonstration of the Issue  
I encountered an issue while implementing **backpropagation** in Elixir. Here’s a short video explaining the problem:  

[![Watch the video](https://img.youtube.com/vi/yX2eMNW09gE/0.jpg)](https://youtu.be/yX2eMNW09gE)  

> Click the thumbnail or [watch the video here](https://youtu.be/yX2eMNW09gE).  

### ✅ Solution  
To resolve the issue, I had to change the way I **updated the weights** in the functional paradigm of Elixir. Unlike Python, where assignments are **mutable**, Elixir requires explicitly **returning new values** instead of modifying them in place.

---

## 🧠 Neural Networks: A Brief Introduction  

A **Neural Network** is a computational model inspired by the human brain, designed to recognize patterns and solve complex problems. It consists of:  

- **Input Layer** – Receives raw data.
- **Hidden Layers** – Processes the data using weighted connections.
- **Output Layer** – Produces the final prediction.  

### ⚙️ Backpropagation  
**Backpropagation** is the method used to adjust the weights of a neural network based on the error in its predictions. This is done through **gradient descent** and the **chain rule of calculus**.  

In this project, I implemented backpropagation using **Elixir’s Nx** and ran into functional programming challenges that I had to overcome.

---

## 🔧 Technologies Used  

- **Elixir** – Functional programming language
- **Nx** – Numerical computing library for Elixir
- **Python** – Used for initial reference implementation
- **Matplotlib** – For visualizing results in Python  

---

## 📝 Future Improvements  

✅ Add more activation functions (ReLU, Leaky ReLU)  
✅ Implement deeper networks with multiple layers  
✅ Optimize weight updates for performance  
✅ Add unit tests for all major components  

---

## 📩 Contributing  

Want to contribute? Feel free to fork this repository and submit a pull request with your improvements!  

