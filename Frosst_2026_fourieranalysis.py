# -*- coding: utf-8 -*-
"""

@file         patternSpeed.py

@author       Walter Dehnen, Marcin Semczuk
@modified by  Matthew Frosst (2026)  

@copyright    Walter Dehnen, Marcin Semczuk (2022)

@license      GNU GENERAL PUBLIC LICENSE version 3.0
              see file LICENSE for details

@version      0.1   jun-2022 MS  initial code
@version      0.2   sep-2022 WD  patternSpeed.py
@version      0.3   apr-2026 MF  updates for colibre

"""
version = '0.3'

import numpy as np
import warnings
from Dehnen_2022_variance import variance

def window(Q):
    """compute W(Q) = 2(1-Q)²(1+2Q)"""
    Q1 = 1-Q
    return 2*Q1*Q1*(1+Q+Q)

def windowDeriv(Q):
    """compute W(Q) = 2(1-Q)²(1+2Q) and dW/dQ = -12 Q(1-Q)"""
    Q1 = 1-Q
    return 2*Q1*Q1*(1+Q+Q), -12*Q*Q1

def atan(sin,cos):
    """arctan(sin/cos) in the range [0,2π]"""
    psi = np.arctan2(sin,cos)
    return np.where(psi<0, psi+2*np.pi, psi)

def amplPhase3(x, m=2):
    """compute amplitude and phase as functions of cos and sin, also Jacobian"""
    f = np.empty((2))
    J = np.empty((2,3))
    Z = np.hypot(x[1],x[2])
    f[0] = Z/x[0]               # A = √(Cm²+Sm²)/C0
    f[1] = atan(x[2],x[1])/m    # ψ = 1/m atan(Sm/Cm)
    J[0,0] =-f[0]/x[0]          # ∂A/∂C0 = -A/C0
    J[0,1] = x[1]/(Z*x[0])      # ∂A/∂Cm = Cm/√(Cm² + Sm²)/C0
    J[0,2] = x[2]/(Z*x[0])      # ∂A/∂Sm = Sm/√(Cm² + Sm²)/C0
    J[1,0] = 0.0                # ∂ψ/∂C0 = 0
    J[1,1] =-x[2]/(m*Z*Z)       # ∂ψ/∂Cm =-Sm/(Cm² + Sm²)/m
    J[1,2] = x[1]/(m*Z*Z)       # ∂ψ/∂Sm = Cm/(Cm² + Sm²)/m
    return f,J

def amplPhase2(x, C0, m=2):
    f,J = amplPhase3((C0,x[0],x[1]), m=m)
    return f,J[1:,:]

def phaseOmega(x, m=2):
    """compute ψ and Ω=dψ/dt from <cosmφ> <sinmφ> and their time derivatives"""
    f = np.empty((2))
    J = np.empty((2,4))
    im = 1/m
    iQ = 1/(x[0]*x[0]+x[1]*x[1])         # 1/(C²+S²)
    J[0,0] =-im*x[1]*iQ                  # ∂ψ/∂C =-S/(C²+S²)/m
    J[0,1] = im*x[0]*iQ                  # ∂ψ/∂S = C/(C²+S²)/m
    J[0,2] = 0
    J[0,3] = 0
    f[0]   = im*atan(x[1],x[0])          # ψ      = 1/m atan(S/C)
    f[1]   = J[0,0]*x[2] + J[0,1]*x[3]   # Ω = dψ = (C dS - S dC)/(C²+S²)/m
    J[1,0] =( im*x[3] - 2*x[0]*f[1])*iQ  # ∂Ω/∂C  = ( dS/m - 2 C Ω)/(C²+S²)
    J[1,1] =(-im*x[2] - 2*x[1]*f[1])*iQ  # ∂Ω/∂S  = (-dC/m - 2 S Ω)/(C²+S²)
    J[1,2] = J[0,0]                      # ∂Ω/∂dC = ∂ψ/∂C
    J[1,3] = J[0,1]                      # ∂Ω/∂dS = ∂ψ/∂S
    return f,J
        
class FourierMethodFast:
    """
    Stripped-down version of FourierMethod for large simulation suites.

    Key differences from FourierMethod:
    - No sorting of particle arrays (saves O(N log N) per galaxy)
    - Bins are defined externally as fixed radial edges (shared across galaxies)
    - analyseBins() is separated so results can be stored before bar-finding
    - Particles are selected per-bin via boolean masks, not index slices
    """

    def __init__(self, m, x, y, vx, vy, checkFinite=False):
        """
        Compute per-particle Fourier quantities. No sorting performed.

        Parameters: same as FourierMethod.__init__
        """
        asarray = np.asarray_chkfinite if checkFinite else np.asarray
        m  = asarray(m)
        x  = asarray(x)
        y  = asarray(y)
        vx = asarray(vx)
        vy = asarray(vy)

        self.Rq  = x*x + y*y                  # R²  (unsorted)
        iRq      = 1.0 / self.Rq              # reciporacal of R²  (unsorted)
        self.dRq = 2.0 * (x*vx + y*vy)        # dR²/dt
        self.dPh = iRq  * (x*vy - y*vx)       # dφ/dt
        self.mC2 = m * iRq * (x*x - y*y)      # μ cos2φ
        self.mS2 = 2.0 * m * iRq * x * y      # μ sin2φ
        self.m   = m                          # mass

    def analyseBins(self, bin_edges, tophat=True):
        """
        Compute A2, ψ2 and uncertainties for each radial bin.

        Parameters:
        -----------
        bin_edges : 1-D array-like of float
            Monotonically increasing bin-edge radii (same units as x, y).
            Defines N-1 bins for N edges.
        tophat : bool
            Use top-hat weighting. Default: True (recommended, unbiased).

        Returns:
        --------
        bins     : list of boolean arrays, one mask per bin
        binData  : np.ndarray, shape (n_bins, 9)  — same columns as
                   FourierMethod.analyseBins()
        """
        bin_edges = np.asarray(bin_edges, dtype=float)
        if bin_edges.ndim != 1 or len(bin_edges) < 2:
            raise Exception("bin_edges must be 1-D with at least 2 entries")
        if np.any(np.diff(bin_edges) <= 0):
            raise Exception("bin_edges must be strictly monotonically increasing")

        R        = np.sqrt(self.Rq)           # actual radius, computed once
        nP       = len(self.Rq)
        n_bins   = len(bin_edges) - 1
        binData  = np.full((n_bins, 9), np.nan)
        
        for k in range(n_bins):
            lo, hi  = bin_edges[k], bin_edges[k+1]
            mask    = (R >= lo) & (R < hi)
            nB      = int(mask.sum())

            print(lo, hi)
            if nB < 2:                        # nothing useful in this bin
                print("Too few particles in bin ", k, ", found ", nB)
                continue
            else:
                print("N bin ", nB)
            
            Rq_bin  = self.Rq[mask]
            Rq0     = lo * lo
            Rq1     = hi * hi
            Rqm     = 0.5 * (Rq0 + Rq1)

            c0 = self.m   if self.m.ndim == 0 else self.m[mask]
            c2 = self.mC2[mask]
            s2 = self.mS2[mask]
            iD = 1.0 / (Rq1 - Rq0)

            if not tophat:
                iD = 2.0 * iD
                q  = np.abs(Rq_bin - Rqm) * iD
                W  = window(q)
                c0 = W * c0
                c2 = W * c2
                s2 = W * s2

            fac = nP * iD / (2 * np.pi)

            if isinstance(c0, np.ndarray):
                CCS = variance([c0, c2, s2])
                CCS.scale(fac)
                Sd0 = CCS.mean(0)
                AP2 = CCS.propagate(amplPhase3)
            else:
                CS2 = variance([c2, s2])
                CS2.scale(fac)
                Sd0 = float(c0) * fac
                AP2 = CS2.propagate(amplPhase2, args=(Sd0,))

            R0  = lo
            Rm  = np.sqrt(Rqm)
            R1  = hi

            binData[k, :] = (nB, R0, Rm, R1, Sd0,
                             AP2.mean(0), AP2.std_of_mean(0),
                             AP2.mean(1), AP2.std_of_mean(1))
        return binData

    def findBarRegion(self, R0, R1, A2_prof, psi_prof,
                      minA2Bar=0.2, maxDPsi=10.0, minDexBar=0.2, minNumBar=100000):
        """
        Identify the bar region from binData alone.

        Returns: (b0, b1) bin indices into binData, or (None, None) if no bar found.
        R0 and R1 of the bar region can be recovered as:
          R0  = binData[b0, 1] = R0_bar
          R1  = binData[b1, 3] = R1_bar
        """

        b0  = np.argmax(A2[np.where(R1 < 5)])
        if A2[b0] < minA2Bar:
            return 0, 0

        minA2 = max(minA2Bar, 0.5 * A2[b0])
        psi   = psi_prof - psi_prof[b0]
        psi   = np.where(psi >  0.5*np.pi, psi - np.pi,
                np.where(psi < -0.5*np.pi, psi + np.pi, psi))

        nB    = len(binData)
        b1    = b0
        psimin, psimax = psi[b0], psi[b1]
        width = lambda ps: max(ps, psimax) - min(ps, psimin)
        maxDPsi_rad = maxDPsi * np.pi / 180.0

        w0 = width(psi[b0-1]) if b0 > 0   and A2[b0-1] > minA2 else 2
        w1 = width(psi[b1+1]) if b1+1 < nB and A2[b1+1] > minA2 else 2
        while min(w0, w1) < maxDPsi_rad:
            if w0 < w1:
                b0 -= 1
                psimin = min(psi[b0], psimin)
                psimax = max(psi[b0], psimax)
                w0 = width(psi[b0-1]) if b0 > 0   and A2[b0-1] > minA2 else 2
            else:
                b1 += 1
                psimin = min(psi[b1], psimin)
                psimax = max(psi[b1], psimax)
                w1 = width(psi[b1+1]) if b1+1 < nB and A2[b1+1] > minA2 else 2

        # use binData radii to determine indicies containing the bar
        R0_bar   = R0[b0]   # inner edge of first bar bin
        R1_bar   = R1[b1]   # outer edge of last bar bin
        nBar = binData[b0:b1+1, 0].sum()   # total particle count across bar bins
        
        if nBar < minNumBar or np.log10(R1_bar / R0_bar) < 2 * minDexBar:
            return 0, 0

        return b0, b1

    def measureOmega(self, bar_mask, tophat=False):
        """
        Same as FourierMethod.measureOmega() but takes a boolean mask
        instead of index pair (i0, i1).
        """
        nP  = len(self.Rq)
        nB  = int(bar_mask.sum())
        if nB < 100:
            return 0., 0., 0., 0., 0., 0., 0., 0.

        Rq_bar = self.Rq[bar_mask]
        Rq0    = Rq_bar.min()
        Rq1    = Rq_bar.max()
        Rqm    = np.median(Rq_bar)

        c2  =  self.mC2[bar_mask]
        s2  =  self.mS2[bar_mask]
        dc2 = -2 * self.dPh[bar_mask] * s2
        ds2 =  2 * self.dPh[bar_mask] * c2

        if not tophat:
            Q   = Rq_bar - Rqm
            fac = np.where(Q < 0, 1/(Rq0 - Rqm), 1/(Rq1 - Rqm))
            Q  *= fac
            W, dW = windowDeriv(Q)
            dW   *= fac
            dW   *= self.dRq[bar_mask]
            c2   = W * c2
            s2   = W * s2
            dc2  = W * dc2 + dW * self.mC2[bar_mask]
            ds2  = W * ds2 + dW * self.mS2[bar_mask]

        var = variance([c2, s2, dc2, ds2])
        var = var.propagate(phaseOmega)

        return (np.sqrt(Rq0), np.sqrt(Rqm), np.sqrt(Rq1),
                var.mean(0), var.std_of_mean(0),
                var.mean(1), var.std_of_mean(1),
                var.corr(0, 1))


if __name__ == "__main__":
    # --- Define shared grid once, outside any loop ---
    bin_edges = np.logspace(np.log10(0.5), np.log10(15.0), num=51)  # kpc

    # --- Pass 1: store binData for all galaxies (cheap to resume/checkpoint) ---
    all_binData = {}
    for ihalo in halo_ids:
        tool = FourierMethodFast(m, x, y, vx, vy)
        masks, binData = tool.analyseBins(bin_edges)
        all_binData[ihalo] = binData          # save to disk here if needed
        # tool and masks go out of scope — no large arrays kept in memory

    # --- Pass 2: bar-finding and pattern speed (uses only stored binData) ---
    for ihalo in halo_ids:
        tool   = FourierMethodFast(m, x, y, vx, vy)   # reconstruct (or cache)
        masks, _ = tool.analyseBins(bin_edges)          # masks needed for measureOmega
        bar_mask, bar_bins = tool.findBarRegion(masks, all_binData[ihalo])
        if bar_mask is not None:
            result = tool.measureOmega(bar_mask)
