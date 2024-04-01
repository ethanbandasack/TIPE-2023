## TIPE 2023
### Thème : la ville
# Compression de données et vidéosurveillance

Le modèle de compression est inspiré du standard de compression JPEG : d’abord transformer l’image, qui est initialement représentée sous la forme RGB (rouge, vert, bleu) en luminance et chrominances. La première étape de la compression est avec pertes : les composantes sont ensuite converties dans le domaine « spectral » par transformée en cosinus discrète (une version adaptée de la transformée de Fourier discrète). Cette dernière permet de séparer les informations basses fréquences, très significatives à l’œil humain, et les hautes fréquences, qui représentent les détails de l’image. Une étape de quantification permet ensuite d’assigner aux composantes les moins importantes la valeur 0 ou d’autres entiers faibles.

Côté compression sans perte : elle exploite le concept fondamental d’entropie de Shannon. La redondance créée précédemment (beaucoup d’entiers faibles proches de 0) fait diminuer l’entropie des données de l’image. Cela indique qu’il est possible de représenter l’image autrement, de coder les informations de sorte qu’elles occupent moins de place. J’ai exploré différentes techniques : codage de Huffman, codage par plages (RLE), algorithme Move to Front, transformée de Burrows-Wheeler.

Afin de mieux saisir les concepts inhérents à la compression de données, un maximum de fonctions sont codées à la main, ce qui est beaucoup moins efficace qu’utiliser des bibliothèques comme scipy.


Le code n’a jamais atteint un stade utilisable et est resté à l’état de juxtaposition de fonctions utilisables mais sans interface claire. La transformée en cosinus discrète est bien mise en place, mais la compression sans perte n’a pas beaucoup avancé, et le peu de tests menés n’ont été que très peu concluants.
