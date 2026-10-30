"""Generatoare de control cu defecte cunoscute (control pozitiv pentru CRQI).

RANDU (IBM, anii 1960): x_{k+1} = 65539 x_k mod 2^31; cifra = floor(10 x / 2^31).
    Defect clasic: tripletele consecutive cad pe 15 plane (Knuth, TAOCP vol. 2).
LCG mod 10: x_{k+1} = (1103515245 x_k + 12345) mod 2^31; cifra = x mod 10.
    Utilizarea greșită a biților inferiori („rand() % 10”): ultimul bit are perioada 2,
    deci paritatea cifrelor alternează strict.
"""


def randu_digits(n, seed=1):
    out = [0] * n
    x = seed
    for i in range(n):
        x = (65539 * x) % 2147483648
        out[i] = (10 * x) // 2147483648
    return out


def lcg_mod10_digits(n, seed=1):
    out = [0] * n
    x = seed
    for i in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        out[i] = x % 10
    return out


CONTROLS = {"RANDU (control)": randu_digits, "LCG mod 10 (control)": lcg_mod10_digits}
N_CONTROL_DIGITS = 15_688_140      # același volum de cifre ca o sursă reală (6 fișiere)
