from PIL import Image
import numpy as np
from random import random
# from cmath import phase
from time import time
from matplotlib import pyplot as plt
from scipy import fftpack as ft
import matplotlib.pyplot as plt
from tqdm import tqdm
im = Image.open("Shanghai.jpg")
plt.ioff()

R = (1366**2+768**2)**.5/13.3 # correspond aux dimensions de l'écran


def afficher(img, path, dpi = R):
    
    fig = plt.figure(figsize = (l/dpi, h/dpi))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    plt.savefig(f"{path}/Image{time()}.png", bbox_inches="tight", pad_inches = 0, dpi=dpi)
    plt.show()

def sauv(img, path, dpi = R):
    
    fig = plt.figure(figsize = (l/dpi, h/dpi))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    plt.savefig(f"{path}/Image{time()}.png", bbox_inches="tight", pad_inches = 0, dpi=dpi)

def imgtoarray(img=im, largeur=None, hauteur = None):
    
    """ transforme, de manière assez rudimentaire,
    une image en la matrice de ses pixels,
    représentés par une liste de trois entiers "RGB" associée
    Image, int, int -> (largeur, hauteur, 3) np.array"""
    
    l, h = img.size
    
    if (largeur, hauteur) == (None, None):
        largeur, hauteur = l, h
    M = np.zeros((hauteur, largeur, 3), dtype = int)
    for x in range(min(l, largeur)):
        for y in range(min(h, hauteur)):
            (a,b,c) = img.getpixel((x, y))
            M[y][x] = np.array([a, b, c], dtype=int)
    return M

# print(imgtoarray()[0][0])

A = imgtoarray(im)
h, l = len(A), len(A[0])
# for py in range(h):
#     for px in range(l):
#         s=A[py][px][0]*(.5)+A[py][px][1]*(-.4187)+A[py][px][2]*(-0.0813)+.5
#         A[py][px][0]=s
#         A[py][px][1]=A[py][px][2]=0
# afficher(A)

# YCC = np.array([[.299,.587,.114],[-.1687,-.3312,.5],[.5,-.4187,-.0813]])
# RGB = np.linalg.inv(YCC)


def nouv(largeur, hauteur):
    
    """ crée une nouvelle image toute noire d'une taille donnée
    int, int -> Image """
    
    return np.full((hauteur, largeur, 3), 255, dtype=int)


def rd(largeur, hauteur):
    M = np.zeros((hauteur, largeur, 3))
    for i in range(hauteur):
        for j in range(largeur):
            for p in range(3):
                M[i][j][p] = random()
    return M


def degrade(largeur, hauteur, c1, c2):
    (a, b, c) = c1
    (d, e, f) = c2
    M = np.zeros((largeur, hauteur, 3), dtype = np.uint8)
    for x in range(largeur):
        for y in range(hauteur):
            s = x+y
            r = s/(largeur+hauteur)
            M[x][y][0] = int(a*r + d*(1-r))
            M[x][y][1] = int(b*r + e*(1-r))
            M[x][y][2] = int(c*r + f*(1-r))
    return M


def naif(img=A, taux=25):
    
    """ compresse de manière naïve une image donnée sans modifier sa taille,
    en donnant à chaque bloc de taux pixels de côté la couleur moyenne du bloc initial
    Image, int -> np.array"""
    
    l, h = len(img[0]), len(img)
    lc, hc = (l//taux+2)*taux, (h//taux+2)*taux
    M = np.zeros((hc, lc ,3), dtype=int)
    for i in range(h):
        for j in range(l):
            for c in range(3):
                M[i][j][c] = img[i][j][c]
    S = np.zeros((h, l, 3), dtype=int)
    for i in range(taux):
        for j in range(taux):
            S += M[i:h+i,j:l+j]
    S //= taux**2
    comp = np.zeros((h, l, 3), dtype = int)
    for x in tqdm(range(h//taux)):
        for i in range(taux):
            for y in range(l//taux):
                for j in range(taux):
                    comp[x*taux+i][y*taux+j] = S[x*taux][y*taux]
    return comp

# D = Image.fromarray(degrade(500, 500, (255, 0, 0), (0, 255, 255)), "RGB")
# naif()

# afficher(im)
# C = naif()

# for i in tqdm([1, 2, 5, 10, 25, 50]):
            # sauv(naif(A, i))



def reduction(img=im, taux=1):
    
    """ compresse de manière naïve une image donnée
    en changeant sa taille
    Image, int -> Image"""
    
    l, h = img.size
    lc, hc = (l//taux+2)*taux, (h//taux+2)*taux
    M = imgtoarray(img, lc, hc)
    S = np.zeros((h, l, 3))
    for i in range(taux):
        for j in range(taux):
            S += M[i:h+i,j:l+j]
    S /= taux**2
    
    comp = np.zeros((h//taux, l//taux, 3))
    for x in range(h//taux):
        for y in range(l//taux):
            comp[x][y] = S[x*taux][y*taux]
    sauv(comp)
    return comp

def rouge(M):
    h,l = len(M), len(M[0])
    L = M.copy()
    for i in range(h):
        for j in range(l):
            L[i][j][1] = L[i][j][2] = 0
    return L

def vert(M):
    h,l = len(M), len(M[0])
    L = M.copy()
    for i in range(h):
        for j in range(l):
            L[i][j][0] = L[i][j][2] = 0
    return L

def bleu(M):
    h,l = len(M), len(M[0])
    L = M.copy()
    for i in range(h):
        for j in range(l):
            L[i][j][1] = L[i][j][0] = 0
    return L

# R, G, B = rouge(A), vert(A), bleu(A)
# sauv(R)
# sauv(G)
# sauv(B)
# T = naif(R)
# A2 = T+G+B
# sauv(A2)
# T = naif(G)
# A2 = T+R+B
# sauv(A2)
# T = naif(B)
# A2 = T+R+G
# sauv(A2)


def triple(M):
    h,l = len(M), len(M[0])
    T = np.zeros((h, l, 3), dtype = int)
    for y in range(h):
        for x in range(l):
            T[y][x][0] = T[y][x][1] = T[y][x][2] = M[y][x]
    return T

def matY(M):
    h,l = len(M), len(M[0])
    L = np.zeros((h, l),dtype=int)
    coef = np.array([[.299], [.587], [.114]])
    for i in range(h):
        for j in range(l):
            L[i][j] = int(np.dot(M[i][j],coef))
    return L

def matCb(M):
    h,l = len(M), len(M[0])
    L = np.zeros((h, l),dtype=int)
    coef = np.array([[-.1687], [-.3313], [.5]])
    for i in range(h):
        for j in range(l):
            L[i][j] = int(np.dot(M[i][j],coef))
    return L

def matCr(M):
    h,l = len(M), len(M[0])
    L = np.zeros((h, l),dtype=int)
    coef = np.array([[.5], [-.4187], [-.0813]])
    for i in range(h):
        for j in range(l):
            L[i][j] = int(np.dot(M[i][j],coef))
    return L

def divise(M, taille = 8):
    h,l = len(M), len(M[0])
    hr, lr = h//taille, l//taille
    blocs = [[] for _ in range(hr)]
    for y in tqdm(range(hr)):
        for x in range(lr):
            blocs[y].append(M[taille*y:taille*(y+1), taille*x:taille*(x+1)])
    return blocs

def rassemble(blocs):
    taille = len(blocs[0][0])
    hr, lr = len(blocs), len(blocs[0])
    h, l = hr*taille, lr*taille
    M = np.full((h,l), blocs[0][0][0][0])
    for y in tqdm(range(h)):
        for x in range(l):
            M[y][x] = blocs[y//taille][x//taille][y%taille][x%taille]
    return M


def pos(x):
    if x<0:
        return 0
    return x

def lum(c):
    return (2*abs(c[0])+7*abs(c[1])+abs(c[2]))/10

def diff(img=im):
    (l, h) = img.size
    M = imgtoarray(img, l+1, h)[:,1:]
    D = imgtoarray(img)
    D = abs(D - M)
    return D

def gris(img=diff(im)):
    h, l = len(img), len(img[0]) 
    G = np.zeros((h, l, 3))
    for x in range(l):
        for y in range(h):
            lu = lum(img[y][x])
            G[y][x] = (lu, lu, lu)
    return G

def YCbCr(c):
    r, g, b = c[0], c[1], c[2]
    Y = int(0.299*r+.587*g+.114*b)
    Cb = 128 + int(-0.1687*r-0.3313*g+.5*b)
    Cr = 128 + int(0.5*r-0.4187*g-0.0813*b)
    return np.array([Y, Cb, Cr])

def rgb(matY,matCb,matCr):
    h,l = len(matY), len(matY[0])
    chang = np.linalg.inv(np.array([[.299,-.1687,.5],[.587,-.3312,-.4187],[.114,.5,-.0813]]))
    M = np.zeros((h,l,3),dtype=int)
    for y in tqdm(range(h)):
        for x in range(l):
            YCC = np.array([matY[y][x], matCb[y][x], matCr[y][x]])
            couleurs = np.dot(YCC, chang)
            M[y][x][0] = int(couleurs[0])
            M[y][x][1] = int(couleurs[1])
            M[y][x][2] = int(couleurs[2])
    return M

# Y, Cb, Cr = matY(A),matCb(A),matCr(A)
# YY = rgb(Y,np.zeros((h,l)),np.zeros((h,l)))
# BB = rgb(np.zeros((h,l)),Cb,np.zeros((h,l)))
# RR = rgb(np.zeros((h,l)),np.zeros((h,l)),Cr)
# sauv(YY)
# sauv(BB)
# sauv(RR)
# YYY = naif(YY)
# BBB = naif(BB)
# RRR = naif(RR)
# sauv(rgb(matY(YYY), Cb, Cr))
# sauv(rgb(Y, matCb(BB), matCr(RRR)))
# R, G, B = rouge(A), vert(A), bleu(A)

# for i in {8}:
#     N = naif(G,i)
#     print(i)
#     afficher((N+B+R)/255)


# X = nouv(160, 160)
# Z = np.full((160, 160), 0, dtype = int)
# sauv(rgb(matY(A), Z, Z))
# sauv(rgb(Z, matCb(A), matCr(A)))

def a255(M):
    I, J, K = len(M), len(M[0]), len(M[0][0])
    A = np.zeros((I, J, K), dtype = np.uint8)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                A[i][j][k] = int(255*M[i][j][k])
    return A

def inttobin(M):
    h, l = len(M), len(M[0])
    A = ''
    for x in range(h):
        for y in range(l):
            for c in range(3):
                b = bin(M[x][y][c])[2:]
                z = 8 - len(b)
                A += z*'0' + b
    return h, l, A

def inttohex(M):
    h, l = len(M), len(M[0])
    A = ''
    for x in range(h):
        for y in range(l):
            for c in range(3):
                A += hex(M[x][y][c])[2:]
    return h, l, A

def lexico(a, b):
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            return True
    return False

def fusion(L1, L2, comparaison):
    L, M = L1, L2
    l, m = 0, 0
    
    liste = []
    
    while l<len(L) and m<len(M):
        if comparaison(L[l], M[m]):
            liste.append(M[m])
            m += 1
        else:
            liste.append(L[l])
            l += 1
    
    if l == len(L):
        liste += M[m:]
    else:
        liste += L[l:]
    
    return liste          

def merge_sort(L, comparaison):
    
    n = len(L)
    if n == 1:
        return L
    
    return fusion(merge_sort(L[int(n//2):], comparaison), merge_sort(L[:int(n//2)], comparaison), comparaison)

def BWT(L):
    n = len(L)
    M = np.concatenate((L, L))
    permu = [(L,True)] + [(M[i:n+i],False) for i in range(1, n)]
    trie = merge_sort(permu, lexico)
    R = np.full(n, L[0])
    for i in range(n):
        R[i] = trie[i][0][-1]
        if trie[i][1]:
            position = i
    return R, position

def fusion2(L1, L2):
    L, M = L1, L2
    l, m = 0, 0
    
    liste = []
    
    while l<len(L) and m<len(M):
        if L[l][0] > M[m][0]:
            liste.append(M[m])
            m += 1
        else:
            liste.append(L[l])
            l += 1
    
    if l == len(L):
        liste += M[m:]
    else:
        liste += L[l:]
    
    return liste          

def merge_sort2(L):
    
    n = len(L)
    if n == 1:
        return L
    
    return fusion2(merge_sort2(L[int(n//2):]), merge_sort2(L[:int(n//2)]))    

def IBWT(R, position):
    n = len(R)
    T = np.full((n,n), R[0])
    D = np.full((n,n), 0)
    for i in range(n):
        T[i][0] = R[i]
        D[i][0] = R[i]
    P = np.eye(n, k=1)
    for i in range(n):
        T = np.array(merge_sort(list(T), lexico))
        T = np.dot(T, P) + D
    return T[i]


def H(p, a):
    return -sum(a*np.log2(p))

def dft(L):
    n = len(L)
    VF = np.full((n,n), complex(0,0))
    for i in range(n):
        for j in range(n):
            VF[i][j] = complex(np.cos(-i*j*2*np.pi/n), np.sin(-2*np.pi*i*j/n))
    return np.dot(np.dot(VF, L.T), VF.T)

def idft(F):
    n = len(F)
    VF = np.full((n,n), complex(0,0))
    for i in range(n):
        for j in range(n):
            VF[i][j] = complex(np.cos(2*np.pi*i*j/n), np.sin(2*np.pi*i*j/n))
    return np.dot(np.dot(VF, F.T), VF.T)/n**2


# N, M = 3,4
# C = np.zeros((N,N))
# D = np.zeros((M,M))
# s = np.zeros((N,M))
# S = np.zeros((N,M))
# for i in range(N):
#     for j in range(N):
#         C[i][j]=np.cos(np.pi/N*(j+.5)*i)
# for i in range(M):
#     for j in range(M):
#         D[i][j]=np.cos(np.pi/N*(j+.5)*i)

# C *= (2/N)**.5
# D *= (2/N)**.5

# CC = C.copy()
# DD = D.copy()
# for i in range(N):
#     CC[i][0] *= 1/2**.5
# for j in range(M):
#     DD[i][0] *= 1/2**.5

# s1 = np.array([0,1,15,9,1,1])
# S1 = np.zeros(N)
# for k in range(N):
#     for n in range(N):
#         S1[k] += s1[n]*np.cos(np.pi/N*(n+.5)*k)
# print(S1)
# s2 = np.zeros(N)
# for k in range(N):
#     s2[k] += S1[0]/2
#     for n in range(1,N):
#         s2[k] += S1[n]*np.cos(np.pi/N*(k+.5)*n)
# print(s2*2/N)
# import scipy.fftpack as ft
# print(ft.idct(ft.dct(s1))/2/N)
# def coef(x):
#     if x==0:
#         return 1/2**.5
#     else:
#         return 1
# s = np.array([[0,1,1,1],[0,3,1, 1],[0,1,4,1]])
# for i in range(N):
#     for j in range(M):
#         for u in range(N):
#             for v in range(M):
#                 S[u][v] += coef(u)*coef(v)*s[i][j]*np.cos(np.pi/N*(i+.5)*u)*np.cos(np.pi/N*(j+.5)*v)
# S *= 2/(N*M)**.5
# s1 = np.zeros((N,M))
# for u in range(N):
#     for v in range(M):
#         for x in range(N):
#             for y in range(M):
#                 s1[x][y] += coef(u)*coef(v)*S[u][v]*np.cos(np.pi/N*(x+.5)*u)*np.cos(np.pi/N*(y+.5)*v)
# s1 *= 2/(N*M)**.5
# # S = np.dot(CC, np.dot(S, DD.T))
# print(s)
# print(S)
# print(s1)

def dct(s):
    N, M = len(s), len(s[0])
    C = np.zeros((N,N))
    D = np.zeros((M,M))
    for j in range(N):
        C[0][j]=1/2**.5
        for i in range(1,N):
            C[i][j]=np.cos(np.pi/N*(j+.5)*i)
    for j in range(M):
        D[0][j]=1/2**.5
        for i in range(1,M):
            D[i][j]=np.cos(np.pi/M*(j+.5)*i)
    return np.dot(C, np.dot(s,D.T))*2/(N*M)**.5

def idct(S):
    N, M = len(S), len(S[0])
    C = np.zeros((N,N))
    D = np.zeros((M,M))
    for j in range(N):
        C[0][j]=1/2**.5
        for i in range(1,N):
            C[i][j]=np.cos(np.pi/N*(j+.5)*i)
    for j in range(M):
        D[0][j]=1/2**.5
        for i in range(1,M):
            D[i][j]=np.cos(np.pi/M*(j+.5)*i)
    return np.dot(C.T, np.dot(S,D))*2/(N*M)**.5

# B = blocs[0][0]
# BY = matY(B)

# transformes = [[] for _ in range(hauteur)]
# for y in range(hauteur):
#     for x in range(largeur):
#         transformes[y].append(dct(blocs[y][x])//1)
# R = rassemble(transformes)
# afficher(triple(R))

# D = degrade(10,10,(0,0,0),(255,255,255))
# Y = matY(A)
# afficher(triple(Y/255))
# afficher(triple(idct(dct(Y-128))+128)/255)
# blocs = divise(Y)
# hauteur,largeur = len(blocs), len(blocs[0])
# Y -= 128
# D = dct(Y)
# for i in range(8):
#     for j in range(8):
#         if i+j>=7:
#             D[i][j]=0
# ID = idct(D)+128
# afficher((triple(ID))/255)
# L = np.reshape(D, (64))
# m = max(abs(L))
# E = abs(D)/m
# T = triple(E)
# afficher(T)

# print("\hline")
# for i in range(8):
#     t = ""
#     for j in range(7):
#         t += str(round(D[i][j]))
#         t += "&"
#     t += str(round(D[i][7]))
#     u = ""
#     for i in range(len(t)):
#         if t[i]==".":
#             u += ","
#         else:
#             u += t[i]
#     u+="\\\\"
#     print(u)
#     print("\hline")



def tcdY(M):
    blocs = divise(M)
    hc, lc = len(blocs), len(blocs[0])
    transforme = [[] for _ in range(hc)]
    # F = np.ones((8, 8), dtype = int)
    # for i in range(8):
    #     for j in range(8):
    #         if i+j>=5 and (i,j)!=(5,0):
    #             F[i][j]=0
    F = np.array(
[ 16,  11,  10,  16,  24,  40,  51,  61,
  12,  12,  14,  19,  26,  58,  60,  55,
  14,  13,  16,  24,  40,  57,  69,  56,
  14,  17,  22,  29,  51,  87,  80,  62,
  18,  22,  37,  56,  68, 109, 103,  77,
  24,  35,  55,  64,  81, 104, 113,  92,
  49,  64,  78,  87, 103, 121, 120, 101,
  72,  92,  95,  98, 112, 100, 103,  99],dtype=int).reshape((8,8))
    for y in tqdm(range(hc)):
        for x in range(lc):
            transforme[y].append(dct(blocs[y][x]-128)//F)
    return transforme

def tcdC(M):
    blocs = divise(M)
    hc, lc = len(blocs), len(blocs[0])
    transforme = [[] for _ in range(hc)]
    # F = np.ones((8, 8), dtype = int)
    # for i in range(8):
    #     for j in range(8):
    #         if i+j>=5:
    #             F[i][j]=0
    F = np.array(
[ 17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99],dtype=int).reshape((8,8))
    F = np.ones((8,8),dtype=int)
    for y in tqdm(range(hc)):
        for x in range(lc):
            transforme[y].append(dct(blocs[y][x])//F)
    return transforme

def tcdiY(transforme):
    hc, lc = len(transforme), len(transforme[0])
    blocs = [[] for _ in range(hc)]
    F = np.array(
[ 16,  11,  10,  16,  24,  40,  51,  61,
  12,  12,  14,  19,  26,  58,  60,  55,
  14,  13,  16,  24,  40,  57,  69,  56,
  14,  17,  22,  29,  51,  87,  80,  62,
  18,  22,  37,  56,  68, 109, 103,  77,
  24,  35,  55,  64,  81, 104, 113,  92,
  49,  64,  78,  87, 103, 121, 120, 101,
  72,  92,  95,  98, 112, 100, 103,  99],dtype=int).reshape((8,8))
    for y in tqdm(range(hc)):
        for x in range(lc):
            blocs[y].append(idct(transforme[y][x]*F)+128)
    return blocs

def tcdiC(transforme):
    hc, lc = len(transforme), len(transforme[0])
    blocs = [[] for _ in range(hc)]
    F = np.array(
[ 17,  18,  24,  47,  99,  99,  99,  99,
  18,  21,  26,  66,  99,  99,  99,  99,
  24,  26,  56,  99,  99,  99,  99,  99,
  47,  66,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99],dtype=int).reshape((8,8))
    F = np.ones((8,8),dtype=int)
    for y in tqdm(range(hc)):
        for x in range(lc):
            blocs[y].append(idct(transforme[y][x]*F))
    return blocs


def tcos(M):
    Y, Cb, Cr = matY(M), matCb(M), matCr(M)
    Y1, Cb1, Cr1 = tcdiY(tcdY(Y)), tcdiC(tcdC(Cb)), tcdiC(tcdC(Cr))
    # sauv(rgb(rassemble(Y1), Cb, Cr))
    return rgb(rassemble(Y1), rassemble(Cb1), rassemble(Cr1))


sauv(A)
t1 = time() 
print(t1)
sauv(tcos(A))
print(time()-t1)
# Z = np.zeros((h,l), dtype = int)
# Y, Cb, Cr = matY(A),matCb(A),matCr(A)
# YY = rgb(Y,Z,Z)
# BB = rgb(Z,Cb,Z)
# RR = rgb(Z,Z,Cr)
# Y1 = tcdiY(tcdY(Y))
# Y3 = naif(Y,2)
# Y2 = rassemble(Y1)
# sauv(rgb(Y2, Z, Z))
# sauv(rgb(matY(Y3), Z, Z))
# sauv(rgb(Y, matCb(BB), matCr(RRR)))

# Y, Cb, Cr = matY(A),matCb(A),matCr(A)
# unique, counts = np.unique(Y, return_counts=True)
# fig = plt.figure()
# plt.xlabel('Coefficient')
# plt.ylabel("Nombre d'occurrences")
# plt.title("Répartition des coefficients de la DCT de la luminance")
# plt.plot(unique, counts)
# plt.savefig(f"Graphe Y {time()}.png")
# plt.show()

# unique, counts = np.unique(Cb, return_counts=True)
# fig = plt.figure()
# plt.xlabel('Coefficient')
# plt.ylabel("Nombre d'occurrences")
# plt.title("Répartition des coefficients de la DCT de la chrominance bleue")
# plt.plot(unique, counts)
# plt.savefig(f"Graphe Cb {time()}.png")
# plt.show()

# unique, counts = np.unique(Cr, return_counts=True)
# fig = plt.figure()
# plt.xlabel('Coefficient')
# plt.ylabel("Nombre d'occurrences")
# plt.title("Répartition des coefficients de la DCT de la chrominance rouge")
# plt.plot(unique, counts)
# plt.savefig(f"Graphe Cr {time()}.png")
# plt.show()
# IDY = rassemble(tcdiY(tcdY(Y)))
# IDCb = rassemble(tcdiC(tcdC(Cb)))
# IDCr = rassemble(tcdiC(tcdC(Cr)))
# R = (rgb(IDY,IDCb,IDCr))
# afficher(R)

# naif(A,2)
# Z = np.full((160, 160), 0, dtype = int)
# D = matY(A)-128

# h1 = h//8
# l1 = l//8
# D = dct(D)

# for i in range(8):
#     for j in range(8):
#         if i+j>4:
#             D[i][j] = 0
#         else:
#             D[i][j] = abs(D[i][j])/486*255
# # sauv(rgb(D))
# D = idct(D)+128
# for i in range(8):
#     for j in range(8):
#         D[i][j] = max(int((D[i][j])),0)
# sauv(rgb(D, Z, Z))
# print("\hline")
# for i in range(8):
#     t = ""
#     for j in range(7):
#         t += str(round(D[i][j]))
#         t += "&"
#     t += str(round(D[i][7]))
#     u = ""
#     for i in range(len(t)):
#         if t[i]==".":
#             u += ","
#         else:
#             u += t[i]
#     u+="\\\\"
#     print(u)
#     print("\hline")