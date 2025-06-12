<div align="center">

<img loading="lazy" src="https://github.com/Glaubernaoli/PCD---GenomeIdentifier/assets/172425065/bcfc56a4-b124-4988-88b4-e860cb438f27" width=800>

</div>

<h1 align="center"> O mineral com a maior dureza usando Algoritimos Geneticos </h1>

### Tiamat, Turma 2024
###  Glauber Nascimento de Oliveira, Julia Guedes e Lorena Ribeiro  
###  Redes Neurais e Algoritmos Genéticos -  Prof. Dr. Daniel Cassar

 <h4 align="center"> 
     Educational Purpouse
</h4>

<h2 align="left"> 💡 Descrição </h2>

<div align="justify">
Atividade final da disciplina de Redes Neurais, ministrada pelo Dr Daniel Cassar, cujo objetivo é otimizar um problema de interesse científico utilizando um algortimo genético. Foi escolhido o problema da constituição de mineiras e suas propriedades para que esse apresente maior dureza. Para isso, utilizou-se o dataset "Prediction of Mohs Hardness with Machine Learning" que contêm dois datasets um com minerais reais e suas propriedades, e outro com minerais artificias. Para o desenvolvimento do projeto usamos o dataset de minerais reais e construimos 3 Notebooks, um para cada etapa do projeto.
A construção do algoritmo cosistiu na predição dos valores de mineirais com maior dureza e otimização das features (propriedades do mineral) a partir da evolução do algortimo genético, tendo como saída (target) o valor da dureza. Após testes, optamos por utilizar o algoritmo de Knn para predição. Quanto para a estrutura do algorito genético criamos os candidatos a partir do range das features e utilizamos a função objetivo com os valores preditos. Os operadores utilizados foi a seleção por elitismo e por torneio, cruzamento de ponto simples e mutação simples.

O objetivo do projeto era encontrar a combinação das features que apresentasse o maior valor de dureza, sendo estabelecido um limite de valor para 9, visto que esse era a maior dureza obtida nos materiais artificiais.

</div>

<h2 align="left"> 🏹 Target </h2>

`hardeness`: Dureza dos minerais.

<div align="justify">


</div>


<h2 align="left"> 📔 Notebooks e arquivos do projeto </h2>

<div align="justify">

`Mineral_Dataset_Supplementary_Info.csv`: Dataset usado no trabalho.

`5.2 - Análise dos dados & Cria candidato.ipynb`: Estudo do dataset

`5.2 - Treinamento do modelo - Função objetivo.ipynb`: Treinamento do modelo e Definição da Função Objetivo

`Implementação_AG.ipynb`: Implementação do Algoritmo Genético

</div>

<h2 align="left"> 🤖 Modelo Usado </h2>

<div align="justify">

 `Algoritmo Genético`: Este modelo foi utilizado para otimizar os parametrôs do modelo de Knn, visando econtrar propriedades que conferem um maior valor de dureza para os minerais.

  `k-NN`: Este modelo foi utilizado para predizer a dureza dos materiais.

   `Árvore de Decisão`: Este modelo foi utilizado para predizer a dureza dos materiais. (teste)

</div>

<h2 align="left"> 🧰 Métricas Usadas </h2>

<div align="justify">

`RMSE`: métrica utilizada para calcular os erros de previsão do modelo.

</div>

<h2 align="left"> 📁 Acesso ao projeto </h2>

<div align="justify">

Você pode acessar o código pelo github ou, preferencialmente, baixá-lo.

</div>

<h2 align="left"> 🛠️ Abrir e rodar o projeto </h2>

<div align="justify">

Depois de baixar o projeto você deve abrí-lo no Jupyter Notebook/VS code

</div>

<h2 align="left"> 📓 Linguagens e programas usados </h2>

<div align="justify">

`Python`, `Jupyter Notebook`, `VS Code`, `Math`, `Scikit Learn`, `Numpy`, `Pandas` , `Matplotlib`, `Pickel`

</div>

<h2 align="left"> 📖 Referências </h2>

<div align="justify">

1.  [Nguyen, B. et al. Genomic characterization of metastatic patterns from prospective clinical sequencing of 25,000 patients. Cell 185, 563-575.e11 (2022).](https://medium.com/ensina-ai/uma-explica%C3%A7%C3%A3o-visual-para-fun%C3%A7%C3%A3o-de-custo-binary-cross-entropy-ou-log-loss-eaee662c396c)


</div>


<h2 align="center"> :octocat:  Autores </h2>

<div align="center">

|  [<img loading="lazy" src="https://github.com/user-attachments/assets/0c4d1ac3-f05b-499f-8618-bfaf749b3504" width=115><br><sub>Glauber Nascimento</sub>](https://github.com/Glaubernaoli)<br> [<sub>Currículo Lattes</sub>](http://lattes.cnpq.br/0913262665776521)<br> [<sub>Linkedin</sub>](https://www.linkedin.com/in/glauber-naoli/) |  [<img loading="lazy" src="https://github.com/user-attachments/assets/6613216e-1df4-420c-a280-0df1116bfb64" width=115><br><sub>Lorena Ribeiro</sub>](https://github.com/Lorena881)<br> [<sub>Currículo Lattes</sub>](http://lattes.cnpq.br/6324363090286730)<br> [<sub>Linkedin</sub>](https://www.linkedin.com/in/lorena-ribeiro-nascimento-7952a930b/) |  [<img loading="lazy" src="https://github.com/user-attachments/assets/cdd48e73-4c79-4c2e-8321-f07c8abd069c" width=115><br><sub>Julia Guedes</sub>](https://github.com/JuliaGuedesASantos)<br> [<sub>Currículo Lattes</sub>](http://lattes.cnpq.br/1031555112242239)<br> [<sub>Linkedin</sub>](https://www.linkedin.com/in/enzo-januzzi-xavier-9063842b0/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) |
| :---: | :---: | :---: |

<div align="center">


![ILUM, CNPEM, MINISTÉRIO DA EDUCAÇÃO](https://github.com/Glaubernaoli/PCD---GenomeIdentifier/assets/172425065/6c9216ea-0cdb-4dac-aac5-445d505b2804)
