#!/usr/bin/env python3
"""Uniform and accuracy-weighted majority vote on a binary (N, K) correctness matrix.

weighted_mv_accuracy weights slot k by its mean accuracy on the same matrix (in-sample
weights); slot_cv is the coefficient of variation of the per-slot accuracies.
"""
import numpy as np


def mv_accuracy(mat):
    """Uniform majority vote."""
    N, K = mat.shape
    votes = mat.sum(axis=1)
    correct = (votes > K / 2).astype(float)
    if K % 2 == 0:
        correct += 0.5 * (votes == K / 2).astype(float)
    return correct.mean()


def weighted_mv_accuracy(mat):
    """Accuracy-weighted voting: weight each slot by its mean accuracy."""
    N, K = mat.shape
    slot_acc = mat.mean(axis=0)  # per-slot accuracy
    # For each instance, weighted vote = sum(w_k * Y_ik) > 0.5 * sum(w_k)
    weighted_votes = mat @ slot_acc
    threshold = slot_acc.sum() / 2
    # exact ties lose float equality to summation order; real margins are multiples of 1/N, far above 1e-9
    tie = np.isclose(weighted_votes, threshold, rtol=0, atol=1e-9)
    correct = ((weighted_votes > threshold) & ~tie).astype(float)
    correct += 0.5 * tie.astype(float)
    return correct.mean()


def pruned_mv_accuracy(mat, n_drop=2):
    """Slot-pruned MV: drop n_drop worst-accuracy slots, MV on rest."""
    N, K = mat.shape
    slot_acc = mat.mean(axis=0)
    keep = np.argsort(slot_acc)[n_drop:]  # drop worst n_drop
    mat_pruned = mat[:, keep]
    return mv_accuracy(mat_pruned)


def slot_cv(mat):
    """Coefficient of variation of per-slot accuracy."""
    slot_acc = mat.mean(axis=0)
    return slot_acc.std() / slot_acc.mean() if slot_acc.mean() > 0 else 0
