# Autograd Engine -> Transformer (from scratch)

A reverse-mode automatic differentiation engine and a GPT-style transformer,
built from scratch in Python with **NumPy as the only heavy dependency** —
no PyTorch, TensorFlow, or JAX autograd.

## Why this project
To understand backpropagation deeply enough to derive it by hand: a dynamic
computation graph, topological-sort backward pass, and every gradient validated
against finite-difference gradient checking.

## Status
Milestone 0 complete: repository scaffolding and test harness.

## Tech
Python | NumPy | pytest | (Streamlit demo to come)
