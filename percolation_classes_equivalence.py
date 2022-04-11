# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 08:28:01 2022

@author: Thibaut de Saivre

Coloriage des cases de la même classe d'équivalence avec la même couleur

=> image de float
0. correspond à un mur


OPTIMISATION :
lorsqu'on change la couleur, au lieu de regarder la hauteur, il faudrait
regarder la taille de l'arbre. Visuellement, ça serait plus cohérent. Ca veut
dire qu'il faut créer encore une liste pour stocker en entier la taille de chaque
arbre (même manière que pour stocker des classes d'équivalence)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep

from typing import List

limit = 0.2

class PercolationAnimation:
    
    def __init__(self):
        
        self.iteration = 0  # donne l'index suivant dans liste_ordre
        self.dimension = 0  # côté du carré
        self.liste_ordres = np.array([])
        self.image = np.array([[]], dtype=float)  # Image à afficher
        self.graphe = np.array([])  # Graphe des représentants
        
        self.couleurs: List[float] = []
        self.classes_equivalence: List[List[List[int]]]
        
        self.anim: animation = None
        self.frames = []
        
    @property
    def taille(self):
        return self.dimension * self.dimension
    
    @staticmethod
    def add_frame(frames, frame, show: bool=False):
        frames.append([plt.imshow(frame, vmin=0., vmax=1., cmap='gist_ncar')])
        if show:
            plt.show()
        sleep(0.02)
        
    def init_all(self, dimension: int):
        
        self.dimension = dimension
        
        self.image = np.zeros((dimension, dimension))
        
        self.liste_ordres = np.array(range(self.taille))
        np.random.shuffle(self.liste_ordres)
        
        # Couleurs aléatoires pour colorier les classes d'équivalence
        self.couleurs = np.arange(limit, 1., 1/self.taille)
        np.random.shuffle(self.couleurs)
        
        self.graphe = np.array(range(self.taille))
        
        # Classes d'équivalences pour faciliterle coloriage
        self.classes_equivalence = [ [self.to_coordonnees(i)] for i in range(self.taille)]
    
    def coordonnees_suivantes(self) -> List[int]:
        
        index = self.liste_ordres[self.iteration]
        
        i, j = index // self.dimension, index % self.dimension
        
        self.iteration += 1
        
        return i, j
    
    def liste_voisins(self, i: int, j: int) -> List[List[int]]:
        
        # Mur : == 0., on ne prend comme voisins que les autres
        voisins = []
        m, n = self.image.shape
        
        if i > 0 and self.image[i - 1, j] > 0.:  # En haut
            voisins.append((i - 1, j))
        
        if j > 0 and self.image[i, j - 1] > 0.:  # A gauche
            voisins.append((i, j - 1))
            
        if i + 1 < m and self.image[i + 1, j] > 0.:  # En bas
            voisins.append((i + 1, j))
            
        if j + 1 < n and self.image[i, j + 1] > 0.:  # A droite
            voisins.append((i, j + 1))
            
        return voisins
    
    def to_index(self, i : int, j: int) -> int:
        return i * self.dimension + j
    
    def to_coordonnees(self, index: int) -> List[int]:
        return index // self.dimension, index % self.dimension
    
    def couleur_unique(self) -> float:
        # pop n'existe pas avec numpy
        couleur = self.couleurs[-1]
        
        self.couleurs = self.couleurs[:-1]
        
        return couleur
    
    def find(self, i: int, j: int) -> List[int]:
        
        hauteur = 0
        index = self.to_index(i, j)
        
        """
        dernier_representant = index  # Case (i, j)
        
        representant = self.graphe[index]  # Représentant de (i, j)
        self.graphe[dernier_representant] = self.graphe[representant]  # pour la réduction de l'arbre
        """
        representant = index
        
        while representant != self.graphe[representant]:
            
            dernier_representant = representant
            representant = self.graphe[representant]
            
            self.graphe[dernier_representant] = self.graphe[representant]

            hauteur += 1
        
        return representant, hauteur  # Attention à prendre [0] pour le repr seul
        
    def union(self, i: int, j: int):
        
        cases = self.liste_voisins(i, j)
        cases.append((i, j))  # Ajout de la case qui vient d'être coloriée
        # En 0, 0, aucune des autres cases adjacentes n'est rajoutée??
        
        representants = [0] * len(cases)
        hauteurs =[0] * len(cases)
        
        # Recherche des représentants et hauteur d'arbres
        for index, (i, j) in enumerate(cases):
            
            representants[index], hauteurs[index] = self.find(i, j)
            
        # Index du maximum : arbre le plus haut
        index = 0
        maximum = hauteurs[0]
        
        for i, hauteur in enumerate(hauteurs):
            if hauteur > maximum:
                maximum = hauteur
                index = i
                
        representant_commun = representants[index]

        couleur = self.image[self.to_coordonnees(representant_commun)]
        
        # Remplacement des représentants:
        for representant in representants:
            
            if representant != representant_commun:
                self.graphe[representant] = representant_commun
                
                # Actualisation de la classe d'équivalence
                self.classes_equivalence[representant_commun].extend(self.classes_equivalence[representant])
                                
                # Coloriage
                for coordonnees in self.classes_equivalence[representant]:
                    self.image[coordonnees] = couleur
                
                # Nettoyage
                self.classes_equivalence[representant] = []
        
    def test_percolation(self) -> bool:
        # Liste des représentants du bord supérieur
        coords_sup = [(0, i) for i in range(self.dimension)]
        coords_inf = [(self.dimension - 1, i) for i in range(self.dimension)]
        
        representants = [self.find(i, j)[0] for i, j in coords_sup]
        for i, j in coords_inf:
            if self.find(i, j)[0] in representants:
                return True
        return False
    
    def run(self):
        """
        A rajouter : colorier les classes d'équivalence de la même couleur 
        
        -> coordonnees suivantes()
        
        -> colorier la case avec couleur_unique()
        
        -> union(i, j)  avec i, j les coordonnées de la case qui vient
        d'être coloriée
        """
        print("Lancement...")
        fig = plt.figure()
        
        while not self.test_percolation():
            
            PercolationAnimation.add_frame(self.frames, self.image, show=False)
            
            i, j = self.coordonnees_suivantes()
            
            # Coloriage de la case
            self.image[i, j] = self.couleur_unique()
            
            self.union(i, j)

        # Dernière frame:
        PercolationAnimation.add_frame(self.frames, self.image, show=False)
            
        print("Création de l'animation...")
        self.anim = animation.ArtistAnimation(fig, self.frames, interval=200, blit=True, repeat_delay=2000)         
            
        print("Sauvegarde de l'animation")
        self.anim.save(r"C:\Users\Thibaut de Saivre\Desktop\test.mp4")
        
        print("animation sauvegardée")
        
if __name__ == "__main__":
    
    p = PercolationAnimation()
    
    p.init_all(20)
    
    p.run()
    
    plt.imshow(p.image, vmin=0., vmax=1., cmap='gist_ncar')  # cmap de base est très bien
    
    plt.show()