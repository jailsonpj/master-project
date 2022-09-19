#!/usr/bin/env python3

import pandas as pd
import numpy as np

class DatasetCustom:
    def __init__(self, path):
        self.path = path
        self.list_filenames = ["news-aggregator.csv", "news-february.csv", "news-july.csv"]
        self.df_list = list()

    def read_file(self, filename: str) -> pd.DataFrame:
        """
        Metodo para ler um arquivo CSV.

        Parametros
        ----------
        filename (str):
            Nome do arquivo CSV.

        Retornos
        ----------
        pd.DataFrame:
            Dataframe com os dados que estavam no CSV.
        """
        return pd.read_csv(self.path + filename)

    def union_title_corpus(self, df: pd.DataFrame, column_title: str, column_corpus: str) -> pd.DataFrame:
        """
        Metodo responsavel por unir a coluna referente ao Titulo do artigo e a coluna com o corpo do artigo

        Parametros
        ----------
        df (pd.DataFrame):
            Dataframe contendo os dados de artigos lido atraves do arquivo CSV.
        column_title (str):
            Nome da coluna que contem o titulo do arquivo.
        column_corpus (str):
            Nome da coluna que contem o corpus de texto do artigo.

        Retornos
        ----------
        pd.Dataframe:
            Dataframe contendo uma nova coluna (text) com a uniao das colunas column_title e column_corpus.
        """

        df["text"] = df[[column_title, column_corpus]].apply(
            lambda x: str(x[0]) + " " + str(x[1]), axis=1
        )

        return df

    def df_label_encoded(self, df: pd.DataFrame, column_label: str) -> pd.DataFrame:
        """
        Metodo responsavel pela codificacao da coluna de label dos dataframe.

        Parametros
        ----------
        df (pd.DataFrame):
            Dataframe contendo os dados com a coluna a ser codificada.
        column_label (str):
            Nome da coluna de label do dataframe.

        Retornos:
        ----------
        tuple:
            Tupla contendo o dataframe normal (posicao 0) e outro dataframe contendo
            a columna column_label codificada. Nesse contexto, foram considerados
            somentes os valores Left e Right para a coluna a ser codificada.
        """
        
        df = df.query(f"`{column_label}` == 'left' or `{column_label}` == 'right'")
        
        df[column_label] = pd.Categorical(df[column_label])
        df[column_label] = df[column_label].cat.codes
        
        return df.reset_index(drop=True)

    def drop_return(self, df: pd.DataFrame, index: int) -> tuple:
        """
        
        """

        row = df.loc[index]
        df.drop(index, inplace=True)

        return row, df

    def df_partition_label(self, df: pd.DataFrame, column_label: str) -> pd.DataFrame:
        """
        
        """
        df_left = df.query(f"`{column_label}` == 0").reset_index(drop=True)
        df_right = df.query(f"`{column_label}` == 1").reset_index(drop=True)

        MAX = len(df_left) if len(df_left) < len(df_right) else len(df_right)

        df_left, df_right = df_left[:MAX], df_right[:MAX]

        df_all = pd.DataFrame(columns=df.columns)
        
        for idx in range(0, MAX):
            row_left, df_left = self.drop_return(df_left, idx)
            row_right, df_right = self.drop_return(df_right, idx)

            df_all = df_all.append(row_left.to_dict(), ignore_index=True)
            df_all = df_all.append(row_right.to_dict(), ignore_index=True)
        
        return df_all

    def rename_columns(self, df: pd.DataFrame, dict_map: dict) -> pd.DataFrame:
        """
        Metodo responsavel por renomeiar colunas de um dataframe.

        Parametros
        ----------
        df (pd.DataFrame):
            Dataframe contendo as colunas a serem renomeadas.
        dict_map (dict):
            Dicionario contendo o nome das colunas a serem
            renomeiadas, bem como o novo nome a ser
            definido. Exemplo {"old_name": "new_name"}.

        Retornos
        ---------
        pd.DataFrame:
            Dataframe com as colunas renoemadas.
        """

        return df.rename(columns=dict_map)

    def get_dataset_custom(self) -> pd.DataFrame:
        """
        Metodo responsavel por retornar os dados no formarto de um dataframe,
        contendo todos os processos de padronizacao.

        Parametros
        ----------
        None

        Retornos
        ---------
        pd.DataFrame:
            Dataframe contendo os dados a serem utilizados para treinamento
            dos modelos de ML.
        """

        dict_map = {"Bias": "labels"}

        for filename in self.list_filenames:
            self.df_list.append(
                self.read_file(filename)
            )

        df = pd.concat(self.df_list, ignore_index=True)
        df = self.union_title_corpus(df, "Title", "Content")
        df = self.df_label_encoded(df, "Bias")
        df = self.rename_columns(df, dict_map)

        return df

        
