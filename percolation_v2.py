# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:23:05 2022

@author: Thibaut de Saivre


Plus propre :
-> Recherche percolation
-> Animation pour trouver le plus court chemin

Python : tout est passé par référence, mais certains objets sont
mutables ou immutables. 

Normalement, les objets (classes) sont immutables
Pour changer ça: @dataclass crée des objets immutables


Structure pour que ça marche (pour l'animation)
-> une fonction qui crée l'animation
=> des fonctions utilitaires

-> percolation : classe avec les attributs qui vont bien.
ça permettra de stocker les données

-> animation

on peut utiliser animation.save()
"""
import numpy as np
import numpy.random as rd
from typing import List
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep
import matplotlib

matplotlib.use("TKAgg")


class Percolation:
    
    def __init__(self):
        
        self.show = False
        self.iteration = 0
        self.dimension = 0
        self.liste_ordres = np.array([])
        self.image = np.array([[]], dtype=bool)
        self.graphe = np.array([])

    @property
    def taille(self) -> int:
        return self.dimension * self.dimension

    def init_liste_ordres(self):
        
        self.liste_ordres = np.array(range(self.taille))
        rd.shuffle(self.liste_ordres)
        
    def init_image(self):
        
        self.image = np.zeros((self.dimension, self.dimension), dtype=bool)

    def init_graphe(self):
        
        self.graphe = np.array(range(self.taille))

    def coordonnees(self, index: int) -> np.ndarray:
        
        return np.array([index // self.dimension, index % self.dimension])

    def init_all(self, dimension: int):
        
        self.dimension = dimension
        self.iteration = 0
        
        self.init_liste_ordres()
        self.init_image()
        self.init_graphe()
        
    def colorie_case(self):
        
        if self.iteration < self.taille:
            
            # pb ici : tuple (x,y) interprété comme (x,x) (2 lignes)
            i, j = self.coordonnees(self.liste_ordres[self.iteration])
            self.image[i, j] = True
    
    def liste_cases_adjacentes(self) -> List:
        
        i, j = self.coordonnees(self.liste_ordres[self.iteration])
        cases = [(i, j)]

        # Case du haut:
        if i > 0:
            cases.append((i - 1, j))
            
        if i + 1 < self.dimension:
            cases.append((i + 1, j))
            
        if j > 0:
            cases.append((i, j - 1))
        
        if j + 1 < self.dimension:
            cases.append((i, j + 1))
            
        return cases

    def liste_cases_a_joindre(self) -> List:
        
        cases = self.liste_cases_adjacentes()
        
        return [case for case in cases if self.image[case]]
    
    def union(self):
        
        cases = self.liste_cases_a_joindre()
        
        representants = []
        longueurs = []
        
        for case in cases:
            representant, longueur = self.find(case)
            representants.append(representant)
            longueurs.append(longueur)
        
        # index du maximum
        index = 0
        maximum = longueurs[0]
        
        for i, val in enumerate(longueurs):
            if val > maximum:
                maximum = val
                index = i
                
        # représentant de l'arbre le plus long
        representant_final = representants[index]
        
        # remplacement des représentants
        for representant in representants:
            self.graphe[representant] = representant_final

    def find(self, coords):

        index = coords[0] * self.dimension + coords[1]
        longueur = 0
        
        dernier_representant = index
        representant = self.graphe[index]
        self.graphe[dernier_representant] = self.graphe[representant]  # pour la réduction de l'arbre
        
        while representant != self.graphe[representant]:
            
            dernier_representant = representant
            representant = self.graphe[representant]
            
            self.graphe[dernier_representant] = self.graphe[representant]

            longueur += 1
        
        return representant, longueur  # Attention à prendre [0] pour le repr seul

    def test_percolation(self) -> bool:
        # Liste des représentants du bord supérieur
        coords_sup = [(0, i) for i in range(self.dimension)]
        coords_inf = [(self.dimension - 1, i) for i in range(self.dimension)]
        
        representants = [self.find(coords)[0] for coords in coords_sup]
        for coords in coords_inf:
            if self.find(coords)[0] in representants:
                return True
        return False

    def next_iteration(self):
        
        # Coloration de la case suivante
        self.colorie_case()
        
        # Jonction des cases adjacentes : si elles sont toutes coloriées :
        # 4 cases à joindre
        self.union()    
        """
        A vérifier :
        si les cases adjacentes existent et sont coloriées (faire une liste de coords)
        on ajoute la case qui vient d'être coloriée
        dans l'ordre : joindre toutes les cases en ajoutant toutes les cases...
        à la racine de l'arbre le plus long
        
        self.liste_cases_a_joindre donne les cases à joindre
        il faut trouver le représentant de chacun et le stocker dans une
        liste, faire la même chose avec la taille, et changer tous les représentants
        pour celui qui a la taille max
        """
        if self.show:
            plt.imshow(self.image)
            plt.show()
        
        self.iteration += 1
        
    def derniere_case(self):
        
        i = self.iteration - 1
        
        return self.coordonnees(self.liste_ordres[i])

    def run(self) -> int:
        
        while not self.test_percolation():
            self.next_iteration()
            
        print("Percolation au bout de", self.iteration, "itérations :", 100 * self.iteration / self.taille, "% de cases coloriées")
        # plt.imshow(self.image)
        # plt.show()
        # sleep(5)
        """
        # recherche du plus court chemin
        # dernière case ajoutée :
        i, j = self.coordonnees(self.liste_ordres[self.iteration - 1])
        
        
        # début du chemin : depuis cette case, vers le haut
        debut = animation_plus_court_chemin(self.image, i, j, True)
        debut.reverse()

        # fin du chemin : depuis cette case, vers le bas
        fin = animation_plus_court_chemin(self.image, i, j, False)
        
        chemin = debut + fin[1:]
        
        affiche_chemin(self.image, chemin)
        """
        return self.iteration

    def moyenne(self, dimension: int, n: int):
        
        val = 0
        
        for i in range(n):
            
            self.init_all(dimension)
            
            val += self.run()
            
        print("Pourcentage moyen pour", n, "itérations :", val / (n * dimension * dimension))


class AnimationChemin:

    def __init__(self, image: np.ndarray, i: int, j: int):

        self.image = image
        self.milieu = (i, j)
        
        self.anim: animation = None  # Stockage de l'animation
    
    # Fonctions auxiliaires
    @staticmethod
    def points_adjacents(image, i, j, limite=0.01) -> List:

        m, n = image.shape
        L = []
        # A gauche
        if j > 0 and image[i, j - 1] < limite:
            L.append((i, j - 1))
        
        # En haut
        if i > 0 and image[i - 1, j] < limite:
            L.append((i - 1, j))
            
        # A droite
        if j + 1 < n and image[i, j + 1] < limite:
            L.append((i, j + 1))
        
        # En bas
        if i + 1 < m and image[i + 1, j] < limite:
            L.append((i + 1, j))

        # Marquage des cases pour pas qu'elles ne soient ajoutées à nouveau à la queue
        for coord in L:
            image[coord] = 0.01
        return L

    @staticmethod
    def add_frame(frames, frame):
        frames.append([plt.imshow(frame)])
        sleep(0.02)

    def create_animation(self) -> animation:
        """
        Création de l'animation
        
        i, j : goulot d'étranglement
        
        matrice :
        0. : vide
        1. : mur
        
        0.8 : chemin (ne pas repasser dessus !)
        0.3 : cases cherchées

        On pourrait tout faire d'un coup :
        recherche d'un chemin, et si jamais on arrive en haut ou en bas, on trace le chemin, puis on
        continue
        """
        frames = []  # Frames à afficher
        fig = plt.figure()
        image = np.ones(self.image.shape) - np.array(self.image, dtype=float)
        queue = [(self.milieu, [])]

        haut_atteint = False
        bas_atteint = False
        bas = image.shape[0] - 1

        while not haut_atteint or not bas_atteint:

            (i, j), chemin = queue.pop(0)

            # Ajout de la frame
            image[i, j] = 0.3  # Fait partie du chemin
            AnimationChemin.add_frame(frames, image)

            if i == 0 and not haut_atteint:  # Le haut est atteint
                chemin.append((i, j))

                # Affichage du chemin
                haut_atteint = True
                for coords in chemin:
                    image[coords] = 0.8
                    AnimationChemin.add_frame(frames, image)

            elif i == bas and not bas_atteint:
                chemin.append((i, j))

                # Affichage du chemin :
                bas_atteint = True
                for coords in chemin:
                    image[coords] = 0.8
                    AnimationChemin.add_frame(frames, image)

            voisins = AnimationChemin.points_adjacents(image, i, j)
            queue.extend([(coords, chemin + [(i, j)]) for coords in voisins])

        print("Création de l'animation")
        # Création de l'animation
        self.anim = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=2000)
        print("Animation créée")

        return self.anim


def test():
    
    p = Percolation()
    p.init_all(30)
    p.run()
    
    i, j = p.derniere_case()
    
    a = AnimationChemin(p.image, i, j)
    
    return a.create_animation()


if __name__ == "__main__":
    print("percolation et création de l'animation...")
    anim = test()
    print("sauvegarde de l'animation...")
    anim.save(r"C:\Users\Thibaut de Saivre\Desktop\test.mp4")
    print("animation sauvegardée")
