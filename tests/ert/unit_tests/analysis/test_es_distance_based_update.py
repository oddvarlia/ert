import numbers
from abc import ABC
from pathlib import Path

import numpy as np
import numpy.typing as npt
import psutil
import pytest
import scipy as sp  # type: ignore
from iterative_ensemble_smoother.esmda_inversion import (
    inversion_exact_cholesky,
    inversion_subspace,
    normalize_alpha,
    singular_values_to_keep,
)
from iterative_ensemble_smoother.utils import sample_mvnormal

from ert.config.field import Field
from ert.field_utils import (
    ErtboxParameters,
    FieldFileFormat,
)


# DistanceESMDA temporary included here
class BaseESMDA(ABC):
    def __init__(
        self,
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        seed: np.random._generator.Generator | int | None = None,
    ) -> None:
        """Initialize the instance."""
        # Validate inputs
        if not (isinstance(covariance, np.ndarray) and covariance.ndim in (1, 2)):
            raise TypeError(
                "Argument `covariance` must be a NumPy array of dimension 1 or 2."
            )

        if covariance.ndim == 2 and covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Argument `covariance` must be square if it's 2D.")

        if not (isinstance(observations, np.ndarray) and observations.ndim == 1):
            raise TypeError("Argument `observations` must be a 1D NumPy array.")

        if not observations.shape[0] == covariance.shape[0]:
            raise ValueError("Shapes of `observations` and `covariance` must match.")

        if not (
            isinstance(seed, (int, np.random._generator.Generator)) or seed is None
        ):
            raise TypeError(
                "Argument `seed` must be an integer "
                "or numpy.random._generator.Generator."
            )

        # Store data
        self.observations = observations
        self.iteration = 0
        self.rng = np.random.default_rng(seed)

        # Only compute the covariance factorization once
        # If it's a full matrix, we gain speedup by only computing cholesky once
        # If it's a diagonal, we gain speedup by never having to compute cholesky
        if isinstance(covariance, np.ndarray) and covariance.ndim == 2:
            self.C_D_L = sp.linalg.cholesky(covariance, lower=False)
        elif isinstance(covariance, np.ndarray) and covariance.ndim == 1:
            self.C_D_L = np.sqrt(covariance)
        else:
            raise TypeError("Argument `covariance` must be 1D or 2D array")

        self.C_D = covariance
        assert isinstance(self.C_D, np.ndarray) and self.C_D.ndim in (1, 2)

    def perturb_observations(
        self, *, ensemble_size: int, alpha: float
    ) -> npt.NDArray[np.double]:
        """Create a matrix D with perturbed observations.

        In the Emerick (2013) paper, the matrix D is defined in section 6.
        See section 2(b) of the ES-MDA algorithm in the paper.

        Parameters
        ----------
        ensemble_size : int
            The ensemble size, i.e., the number of columns in the returned array,
            which is of shape (num_observations, ensemble_size).
        alpha : float
            The covariance inflation factor. The sequence of alphas should
            obey the equation sum_i (1/alpha_i) = 1. However, this is NOT enforced
            in this method call. The user/caller is responsible for this.

        Returns
        -------
        D : np.ndarray
            Each column consists of perturbed observations,
            observation std is scaled by sqrt(alpha).

        """
        # Draw samples from zero-centered multivariate normal with cov=alpha * C_D,
        # and add them to the observations. Notice that
        # if C_D = L @ L.T by the cholesky factorization, then drawing y from
        # a zero cented normal means that y := L @ z, where z ~ norm(0, 1).
        # Therefore, scaling C_D by alpha is equivalent to scaling L with sqrt(alpha).

        D: npt.NDArray[np.double] = self.observations[:, np.newaxis] + np.sqrt(
            alpha
        ) * sample_mvnormal(C_dd_cholesky=self.C_D_L, rng=self.rng, size=ensemble_size)
        assert D.shape == (len(self.observations), ensemble_size)
        return D


class ESMDA(BaseESMDA):
    """
    Implement an Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    The implementation follows :cite:t:`EMERICK2013`.

    Parameters
    ----------
    covariance : np.ndarray
        Either a 1D array of diagonal covariances, or a 2D covariance matrix.
        The shape is either (num_observations,) or (num_observations, num_observations).
        This is C_D in Emerick (2013), and represents observation or measurement
        errors. We observe d from the real world, y from the model g(x), and
        assume that d = y + e, where the error e is multivariate normal with
        covariance given by `covariance`.
    observations : np.ndarray
        1D array of shape (num_observations,) representing real-world observations.
        This is d_obs in Emerick (2013).
    alpha : int or 1D np.ndarray, optional
        Multiplicative factor for the covariance.
        If an integer `alpha` is given, an array with length `alpha` and
        elements `alpha` is constructed. If an 1D array is given, it is
        normalized so sum_i 1/alpha_i = 1 and used. The default is 5, which
        corresponds to np.array([5, 5, 5, 5, 5]).
    seed : integer or numpy.random._generator.Generator, optional
        A seed or numpy.random._generator.Generator used for random number
        generation. The argument is passed to numpy.random.default_rng().
        The default is None.
    inversion : str, optional
        Which inversion method to use. The default is "exact".
        See the dictionary ESMDA._inversion_methods for more information.

    Examples
    --------
    >>> covariance = np.diag([1, 1, 1])
    >>> observations = np.array([1, 2, 3])
    >>> esmda = ESMDA(covariance, observations)

    """

    # Available inversion methods. The inversion methods all compute
    # C_MD @ (C_DD + alpha * C_D)^(-1)  @ (D - Y)
    _inversion_methods = {
        "exact": inversion_exact_cholesky,
        "subspace": inversion_subspace,
    }

    def __init__(
        self,
        covariance: npt.NDArray[np.double],
        observations: npt.NDArray[np.double],
        alpha: int | npt.NDArray[np.double] = 5,
        seed: np.random._generator.Generator | int | None = None,
        inversion: str = "exact",
    ) -> None:
        """Initialize the instance."""

        super().__init__(covariance=covariance, observations=observations, seed=seed)

        if not (
            (isinstance(alpha, np.ndarray) and alpha.ndim == 1)
            or isinstance(alpha, numbers.Integral)
        ):
            raise TypeError("Argument `alpha` must be an integer or a 1D NumPy array.")

        if not isinstance(inversion, str):
            raise TypeError(
                "Argument `inversion` must be a string in "
                f"{tuple(self._inversion_methods.keys())}, but got {inversion}"
            )
        if inversion not in self._inversion_methods.keys():
            raise ValueError(
                "Argument `inversion` must be a string in "
                f"{tuple(self._inversion_methods.keys())}, but got {inversion}"
            )

        # Store data
        self.inversion = inversion

        # Alpha can either be an integer (num iterations) or a list of weights
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            self.alpha = normalize_alpha(alpha)
        elif isinstance(alpha, numbers.Integral):
            self.alpha = normalize_alpha(np.ones(alpha))
            assert np.allclose(self.alpha, normalize_alpha(self.alpha))
        else:
            raise TypeError("Alpha must be integer or 1D array.")

    def num_assimilations(self) -> int:
        return len(self.alpha)

    def assimilate(
        self,
        X: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        *,
        overwrite: bool = False,
        truncation: float = 1.0,
        D: npt.NDArray[np.double] = None,
    ) -> npt.NDArray[np.double]:
        """Assimilate data and return an updated ensemble X_posterior.

            X_posterior = smoother.assimilate(X, Y)

        Parameters
        ----------
        X : np.ndarray
            A 2D array of shape (num_parameters, ensemble_size). Each row
            corresponds to a parameter in the model, and each column corresponds
            to an ensemble member (realization).
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.
        overwrite : bool
            If True, then arguments X and Y may be overwritten and mutated.
            If False, then the method will not mutate inputs in any way.
            Setting this to True might save memory.
        truncation : float
            How large a fraction of the singular values to keep in the inversion
            routine. Must be a float in the range (0, 1]. A lower number means
            a more approximate answer and a slightly faster computation.

        Returns
        -------
        X_posterior : np.ndarray
            2D array of shape (num_parameters, ensemble_size).

        """
        if self.iteration >= self.num_assimilations():
            raise Exception("No more assimilation steps to run.")

        # Verify shapes
        _, num_ensemble = X.shape
        num_outputs, num_emsemble2 = Y.shape
        assert num_ensemble == num_emsemble2, (
            "Number of ensemble members in X and Y must match"
        )
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("Argument `X` must contain floats")
        if not np.issubdtype(Y.dtype, np.floating):
            raise TypeError("Argument `Y` must contain floats")

        assert 0 < truncation <= 1.0

        # Do not overwrite input arguments
        if not overwrite:
            X, Y = np.copy(X), np.copy(Y)

        # Line 2 (b) in the description of ES-MDA in the 2013 Emerick paper

        # Draw samples from zero-centered multivariate normal with cov=alpha * C_D,
        # and add them to the observations. Notice that
        # if C_D = L L.T by the cholesky factorization, then drawing y from
        # a zero cented normal means that y := L @ z, where z ~ norm(0, 1)
        # Therefore, scaling C_D by alpha is equivalent to scaling L with sqrt(alpha)
        if D is None:
            D = self.perturb_observations(
                ensemble_size=num_ensemble, alpha=self.alpha[self.iteration]
            )

        assert D.shape == (num_outputs, num_ensemble)

        # Line 2 (c) in the description of ES-MDA in the 2013 Emerick paper
        # Choose inversion method, e.g. 'exact'. The inversion method computes
        # C_MD @ sp.linalg.inv(C_DD + C_D_alpha) @ (D - Y)
        inversion_func = self._inversion_methods[self.inversion]

        # Update and return
        X += inversion_func(
            alpha=self.alpha[self.iteration],
            C_D=self.C_D,
            D=D,
            Y=Y,
            X=X,
            truncation=truncation,
        )

        self.iteration += 1
        return X

    def compute_transition_matrix(
        self,
        Y: npt.NDArray[np.double],
        *,
        alpha: float,
        truncation: float = 1.0,
    ) -> npt.NDArray[np.double]:
        """Return a matrix T such that X_posterior = X_prior + X_prior @ T.

        The purpose of this method is to facilitate row-by-row, or batch-by-batch,
        updates of X. This is useful if X is too large to fit in memory.

        Parameters
        ----------
        Y : np.ndarray
            2D array of shape (num_observations, ensemble_size), containing
            responses when evaluating the model at X. In other words, Y = g(X),
            where g is the forward model.
        alpha : float
            The covariance inflation factor. The sequence of alphas should
            obey the equation sum_i (1/alpha_i) = 1. However, this is NOT enforced
            in this method call. The user/caller is responsible for this.
        truncation : float
            How large a fraction of the singular values to keep in the inversion
            routine. Must be a float in the range (0, 1]. A lower number means
            a more approximate answer and a slightly faster computation.

        Returns
        -------
        T : np.ndarray
            A matrix T such that X_posterior = X_prior + X_prior @ T.
            It has shape (num_ensemble_members, num_ensemble_members).
        """

        # Recall the update equation:
        # X += C_MD @ (C_DD + alpha * C_D)^(-1)  @ (D - Y)
        # X += X @ center(Y).T / (N-1) @ (C_DD + alpha * C_D)^(-1) @ (D - Y)
        # We form T := center(Y).T / (N-1) @ (C_DD + alpha * C_D)^(-1) @ (D - Y),
        # so that
        # X_new = X_old + X_old @ T
        # or
        # X += X @ T

        D = self.perturb_observations(ensemble_size=Y.shape[1], alpha=alpha)
        inversion_func = self._inversion_methods[self.inversion]
        return inversion_func(
            alpha=alpha,
            C_D=self.C_D,
            D=D,
            Y=Y,
            X=None,  # We don't need X to compute the factor T
            truncation=truncation,
            return_T=True,  # Ensures that we don't need X
        )


class DistanceESMDA(ESMDA):
    def assimilate(
        self,
        *,
        X: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        rho: npt.NDArray[np.double],
        truncation: float = 0.99,
    ):
        """
        Implementation of algorithm described in Appendix B of emerick16a.

        X_posterior = smoother.assimilate(X, Y, D, alpha, rho)

        """

        N_n, N_e = Y.shape

        # Subtract the mean of every parameter, see Eqn (B.4)
        M_delta = X - np.mean(X, axis=1, keepdims=True)

        # Subtract the mean of every observation, see Eqn (B.5)
        D_delta = Y - np.mean(Y, axis=1, keepdims=True)

        # See Eqn (B.8)
        # Compute the diagonal of the inverse of S directly, without forming S itself.
        if self.C_D.ndim == 1:
            # If C_D is 1D, it's a vector of variances. S_inv_diag is 1/sqrt(variances).
            S_inv_diag = 1.0 / np.sqrt(self.C_D)
        else:
            # If C_D is 2D, extract its diagonal of variances, then compute S_inv_diag.
            S_inv_diag = 1.0 / np.sqrt(np.diag(self.C_D))

        # See Eqn (B.10)
        U, w, VT = sp.linalg.svd(S_inv_diag[:, np.newaxis] * D_delta)
        idx = singular_values_to_keep(w, truncation=truncation)
        N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
        U_r, w_r = U[:, :N_r], w[:N_r]

        # See Eqn (B.12)
        # Calculate C_hat_D, the correlation matrix of measurement errors.
        # This is defined as C_hat_D = S^-1 * C_D * S^-1
        if self.C_D.ndim == 1:
            # If C_D is a 1D vector of variances,
            # it represents a diagonal matrix.
            # In this special case,
            # S_inv * C_D * S_inv simplifies to the identity matrix.
            C_hat_D = np.identity(N_n)
        else:  # C_D is a 2D matrix
            # This scales each ROW i of self.C_D by the scalar S_inv_diag[i].
            # This is numerically identical to the matrix multiplication S⁻¹ @ C_D
            C_hat_D_temp = S_inv_diag[:, np.newaxis] * self.C_D

            # This scales each COLUMN j of C_hat_D_temp by the scalar S_inv_diag[j].
            # This is numerically identical to the matrix multiplication (result) @ S⁻¹
            C_hat_D = C_hat_D_temp * S_inv_diag

        U_r_w_inv = U_r / w_r
        # See Eqn (B.13)
        R = (
            self.alpha
            * (N_e - 1)
            * np.linalg.multi_dot([U_r_w_inv.T, C_hat_D, U_r_w_inv])
        )

        # See Eqn (B.14)
        H_r, Z_r = sp.linalg.eigh(R, driver="evr", overwrite_a=True)

        # See Eqn (B.18)
        _X = (S_inv_diag[:, np.newaxis] * U_r) * (1 / w_r) @ Z_r

        # See Eqn (B.19)
        L = np.diag(1.0 / (1.0 + H_r))

        # See Eqn (B.20)
        X1 = L @ _X.T
        # See Eqn (B.21)
        X2 = D_delta.T @ _X
        # See Eqn (B.22)
        X3 = X2 @ X1

        # See Eqn (B.23)
        K_i = M_delta @ X3

        # See Eqn (B.24)
        K_rho_i = rho * K_i

        D = self.perturb_observations(ensemble_size=N_e, alpha=self.alpha)
        # See Eqn (B.25)
        X4 = K_rho_i @ (D - Y)

        # See Eqn (B.26)
        return X + X4

    def prepare_assimilation(
        self,
        Y: npt.NDArray[np.double],
        truncation: float = 0.99,
        D: npt.NDArray[np.double] = None,
    ):
        """
        Implementation of algorithm described in Appendix B of emerick16a.
        The first steps to prepare X3 only depends on Y and observations.
        These steps can be calculated once when running multiple batches of
        field parameters. This function will calculate X3 and perturbed observations D.
        """
        N_n, N_e = Y.shape

        # Subtract the mean of every observation, see Eqn (B.5)
        D_delta = Y - np.mean(Y, axis=1, keepdims=True)

        # See Eqn (B.8)
        # Compute the diagonal of the inverse of S directly, without forming S itself.
        if self.C_D.ndim == 1:
            # If C_D is 1D, it's a vector of variances. S_inv_diag is 1/sqrt(variances).
            S_inv_diag = 1.0 / np.sqrt(self.C_D)
        else:
            # If C_D is 2D, extract its diagonal of variances, then compute S_inv_diag.
            S_inv_diag = 1.0 / np.sqrt(np.diag(self.C_D))

        # See Eqn (B.10)
        U, w, VT = sp.linalg.svd(S_inv_diag[:, np.newaxis] * D_delta)
        idx = singular_values_to_keep(w, truncation=truncation)
        N_r = min(N_n, N_e - 1, idx)  # Number of values in SVD to keep
        U_r, w_r = U[:, :N_r], w[:N_r]

        # See Eqn (B.12)
        # Calculate C_hat_D, the correlation matrix of measurement errors.
        # This is defined as C_hat_D = S^-1 * C_D * S^-1
        if self.C_D.ndim == 1:
            # If C_D is a 1D vector of variances,
            # it represents a diagonal matrix.
            # In this special case,
            # S_inv * C_D * S_inv simplifies to the identity matrix.
            C_hat_D = np.identity(N_n)
        else:  # C_D is a 2D matrix
            # This scales each ROW i of self.C_D by the scalar S_inv_diag[i].
            # This is numerically identical to the matrix multiplication S⁻¹ @ C_D
            C_hat_D_temp = S_inv_diag[:, np.newaxis] * self.C_D

            # This scales each COLUMN j of C_hat_D_temp by the scalar S_inv_diag[j].
            # This is numerically identical to the matrix multiplication (result) @ S⁻¹
            C_hat_D = C_hat_D_temp * S_inv_diag

        U_r_w_inv = U_r / w_r
        # See Eqn (B.13)
        R = (
            self.alpha
            * (N_e - 1)
            * np.linalg.multi_dot([U_r_w_inv.T, C_hat_D, U_r_w_inv])
        )

        # See Eqn (B.14)
        H_r, Z_r = sp.linalg.eigh(R, driver="evr", overwrite_a=True)

        # See Eqn (B.18)
        _X = (S_inv_diag[:, np.newaxis] * U_r) * (1 / w_r) @ Z_r

        # See Eqn (B.19)
        L = np.diag(1.0 / (1.0 + H_r))

        # See Eqn (B.20)
        X1 = L @ _X.T
        # See Eqn (B.21)
        X2 = D_delta.T @ _X

        # The matrices X3 and D with perturbed observations is saved
        # and re-used in assimilation of each batch of field parameters

        # See Eqn (B.22)
        self.X3 = X2 @ X1

        # Observations with added perturbations
        if D is None:
            print("Calculate C_D inside class DistanceESMDA")
            self.D = self.perturb_observations(ensemble_size=N_e, alpha=self.alpha)
        else:
            print("Assign D inside class DistanceESMDA")
            assert self.C_D.shape[0] == D.shape[0]
            self.D = D

    def assimilate_batch(
        self,
        *,
        X_batch: npt.NDArray[np.double],
        Y: npt.NDArray[np.double],
        rho_batch: npt.NDArray[np.double],
    ):
        """
        Implementation of algorithm described in Appendix B of emerick16a.
        Require that prepare_assimilation is run after the DistanceESMDA is created.
        It's main purpose is to avoid recalculating matrices in the algorithm
        that does not change from batch to batch when running batch by batch
        of field parameters through assimilation.

        X_posterior_batch = smoother.assimilate_batch(X_batch, Y, rho_batch)

        """

        # Subtract the mean of every parameter, see Eqn (B.4)
        M_delta = X_batch - np.mean(X_batch, axis=1, keepdims=True)

        # See Eqn (B.23)
        K_i = M_delta @ self.X3

        # See Eqn (B.24)
        K_rho_i = rho_batch * K_i

        # See Eqn (B.25)
        X4 = K_rho_i @ (self.D - Y)

        # See Eqn (B.26)
        return X_batch + X4


def calc_max_number_of_layers_per_batch_for_distance_localization(
    nx: int,
    ny: int,
    nz: int,
    num_obs: int,
    bytes_per_float: int = 8,  # float64 as default here
) -> int:
    """Calculate number of layers from a 3D field parameter that can be updated
    within available memory. Distance-based localization requires two large matrices
    the Kalman gain matrix K and the localization scaling matrix RHO, both have size
    equal to number of field parameter values times number of observations.
    Therefore, a batching algorithm is used where only a subset of parameters
    is used when calculating the Schur product of RHO and K matrix in the update
    algorithm. This function calculates a batch size in number of grid layers
    of field parameter values that can fit into the available memory,
    accounting for a safety margin.

    Derivation of formula:
    ---------------------
    available_memory = (amount of available memory on system) * memory_safety_factor
    required_memory = 2 * num_params * num_obs * bytes_in_float64
    We want (required_memory < available_memory) so:
        max_num_params_per_batch <= available_memory / (2 * num_obs * bytes_in_float64)
        number_of_batches = num_params / max_num_param_per_batch
    The available memory is checked using the `psutil` library, which provides
    information about system memory usage.
    From `psutil` documentation:
    - available:
        the memory that can be given instantly to processes without the
        system going into swap.
        This is calculated by summing different memory values depending
        on the platform and it is supposed to be used to monitor actual
        memory usage in a cross platform fashion.

    Args:
        nx: grid size in I-direction (local x-axis direction)
        ny: grid size in J-direction (local y-axis direction)
        nz: grid size in K-direction (number of layers)
        num_obs: Number of observations

    Returns:
        Number of batches the field parameter must be split into
        to avoid memory problems

    """
    num_params = nx * ny * nz
    num_param_per_layer = nx * ny
    available_memory_in_bytes = psutil.virtual_memory().available
    memory_safety_factor = 0.8

    # Fields are stored with bytes_per_float.
    max_number_of_params_per_batch = min(
        int(
            np.floor(
                (available_memory_in_bytes * memory_safety_factor)
                / (2 * num_obs * bytes_per_float)
            )
        ),
        num_params,
    )

    return int(max_number_of_params_per_batch / num_param_per_layer)


@pytest.fixture
def init_dl_smoother():
    # Initialize data for testing
    # update_3D_field_with_distance_esmda
    seed = 1298278
    nobs = 3
    nreal = 100
    sigma = 0.1
    observations = np.zeros(nobs, dtype=np.float64)
    obs_xpos = np.zeros(nobs, dtype=np.float64)
    obs_ypos = np.zeros(nobs, dtype=np.float64)
    obs_main_range = np.zeros(nobs, dtype=np.float64)
    obs_perp_range = np.zeros(nobs, dtype=np.float64)
    obs_angles = np.zeros(nobs, dtype=np.float64)

    observations[0] = 1.0
    obs_xpos[0] = 0.0
    obs_ypos[0] = 0.0
    obs_main_range[0] = 100.0
    obs_perp_range[0] = 100.0
    obs_angles[0] = 0.0

    observations[1] = 0.0
    obs_xpos[1] = 200.0
    obs_ypos[1] = 150.0
    obs_main_range[1] = 200.0
    obs_perp_range[1] = 100.0
    obs_angles[1] = 45.0

    observations[2] = -1.0
    obs_xpos[2] = 400.0
    obs_ypos[2] = 250.0
    obs_main_range[2] = 400.0
    obs_perp_range[2] = 100.0
    obs_angles[2] = -45.0

    C_D = np.zeros(nobs, dtype=np.float64)
    C_D[:] = sigma * sigma
    Y = np.zeros((nobs, nreal), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for obs_index in range(nobs):
        Y[obs_index, :] = rng.normal(loc=0.0, scale=sigma, size=nreal)
    alpha = np.array([1.0])
    truncation = 0.99
    dist_esmda_smoother = DistanceESMDA(
        covariance=C_D, observations=observations, alpha=alpha, seed=seed
    )
    dist_esmda_smoother.prepare_assimilation(Y=Y, truncation=truncation, D=None)
    # Define field
    xlength = 1000.0
    ylength = 1200.0
    nx = 10
    ny = 12
    nz = 2
    xinc = xlength / nx
    yinc = ylength / ny
    ertbox_params = ErtboxParameters(
        nx=nx,
        ny=ny,
        nz=nz,
        xlength=xlength,
        ylength=ylength,
        xinc=xinc,
        yinc=yinc,
        rotation_angle=0.0,
        origin=(0.0, 0.0),
    )

    field = Field(
        type="field",
        name="MyField",
        forward_init=True,
        update=True,
        ertbox_params=ertbox_params,
        file_format=FieldFileFormat.ROFF,
        output_transformation=None,
        input_transformation=None,
        truncation_min=-5.0,
        truncation_max=5.0,
        forward_init_file="dummy.roff",
        output_file=Path("dummy.roff"),
        grid_file="grid.roff",
    )

    rho_2D = field.calc_rho_for_2d_grid_layer(
        obs_xpos,
        obs_ypos,
        obs_main_range,
        obs_perp_range,
        obs_angles,
        right_handed_grid_indexing=True,
    )

    return {
        "smoother": dist_esmda_smoother,
        "Y": Y,
        "Rho_2D": rho_2D,
        "Field": field,
        "Nreal": nreal,
        "Nobs": nobs,
        "Rng": rng,
    }


def test_calc_max_number_of_layers_per_batch_for_distance_localization():
    nx = 200
    ny = 200
    nz = 100
    numobs = 6000
    bytes_per_float = 8
    nbatch = calc_max_number_of_layers_per_batch_for_distance_localization(
        nx, ny, nz, numobs, bytes_per_float=bytes_per_float
    )
    print(f"nbatch = {nbatch}")


def test_update_3D_field_with_distance_esmda(
    snapshot,
    init_dl_smoother,
    nlayer_per_batch: int,
):
    algorithm = init_dl_smoother["smoother"]
    nreal = init_dl_smoother["Nreal"]
#    nobs = init_dl_smoother["Nobs"]
    field = init_dl_smoother["Field"]
    rho_2D = init_dl_smoother["Rho_2D"]
    Y = init_dl_smoother["Y"]
    rng = init_dl_smoother["Rng"]

    nx = field.ertbox_params.nx
    ny = field.ertbox_params.ny
    nz = field.ertbox_params.nz
    a = 1.0 / nx
    b = 1.0 / ny
    indx = 0
    X_prior = np.zeros((nx * ny * nz, nreal), dtype=np.float64)
    for i in range(nx):
        xpos_param = i * nx
        for j in range(ny):
            ypos_param = j * ny
            trend = a * xpos_param + b * ypos_param
            X_prior[indx, :] = trend + rng.normal(loc=0.0, scale=0.5, size=nreal)
            indx += 1

    print("Run update using DistanceESMDA")
    X_post_3D = field.update_3D_field_with_distance_esmda(
        algorithm, field.name, X_prior, Y, rho_2D, nlayer_per_batch, nx, ny, nz
    )
    print(f"Shape of x_post_3D = {X_post_3D.shape}")
