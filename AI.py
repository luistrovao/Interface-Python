import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from numpy import arange, pi, random, linspace
import matplotlib.cm as cm
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

# import keras
# from keras.models import Sequential
# from keras.layers import Dense

class algoritmos_AI():
    def __init__(self):
        self.tipos = {
            "RNA_2C": self.rede_neural_2camadas,
            "Dec_Tree": self.decision_tree,
        }

    def split_clusters(self, base, n_clusters):

        self.classes = []
        self.df = base

        i = 0
        while i < n_clusters:
            self.classes.append(self.df.loc[self.df['K classes'] == i])
            i += 1

        self.df = self.df.drop('K classes', axis=1)

    def rede_neural_2camadas(self, base, n_layer1, n_layer2, Epochs, BatchSize, n_clusters, entradas, saidas):

        from sklearn.model_selection import train_test_split

        classes = []
        df = base

        algoritmos_AI().split_clusters(base, n_clusters)

        dados = []
        i = 0
        previsoes = []
        resultados = []
        layer1 = n_layer1
        layer2 = n_layer2
        Ep = Epochs
        BS = BatchSize

        while i < n_clusters:
            # dados.append(train_test_split(self.classes[i], test_size=0.2))
            print(dados[0][0].loc[:, entradas])
            print(dados[i][0].iloc[:,0:N_inputs])

            regressor = Sequential()
            regressor.add(Dense(units=layer1, activation='relu', input_dim=39))
            regressor.add(Dense(units=layer2, activation='relu'))
            regressor.add(Dense(units=1, activation='linear'))
            regressor.compile(loss='mean_absolute_percentage_error', optimizer='Adamax',
                              metrics=['mean_absolute_percentage_error'])
            i += 1

            regressor.fit(dados[i][0].loc[:, entradas], dados[i][0].loc[:, saidas], batch_size=BS, epochs=Ep)
            previsoes.append(regressor.predict(dados[i][1].loc[:, entradas]))
            nome_modelo = 'modelo_csm_NN_' + str(i) + '.h5'
            fullname = os.path.join(outdir, nome_modelo)
            regressor.save(fullname)
            resultados.append(regressor.evaluate(dados[i][1].loc[:, entradas], dados[i][1].loc[:, saidas]))

    def decision_tree(self, base, max_dep, n_estim, n_clusters, entradas, saidas):

        self.classes = []
        self.df = base

        i = 0
        while i < n_clusters:
            self.classes.append(self.df.loc[self.df['K classes'] == i])
            i += 1

        self.df = self.df.drop('K classes', axis=1)

        regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_dep),
                                 n_estimators=n_estim, random_state=None)
        dados = []
        i = 0
        previsoes = []
        resultados = []
        modelo = []
        k = 0
        N_inputs = len(entradas)
        N_outputs = len(saidas)
        figuras = []

        while i < n_clusters:
            dados.append(train_test_split(self.classes[i], test_size=0.2))
            modelo.append([0] * N_outputs)
            previsoes.append([0] * N_outputs)
            resultados.append([0] * N_outputs)
            while k < N_outputs:
                modelo[i][k] = regr.fit(dados[i][0].loc[:, entradas], dados[i][0].loc[:, saidas[k]])
                previsoes[i][k] = regr.predict(dados[i][1].loc[:, entradas])
                resultados[i][k] = np.mean(100 * abs(np.asarray(dados[i][1].loc[:, saidas[k]]) -
                                                     previsoes[i][k]) / np.asarray(dados[i][1].loc[:, saidas[k]]))

                k += 1
            k = 0
            i += 1


        return dados, previsoes, resultados


class filtros():
    def __init__(self):
        self.funcoes = {
            "nao_numerico": self.nao_num,
            "quartiles": self.quartiles,
            "clusterizacao": self.clusters,
        }

    def nao_num(self, base):
        # Converte para Dataframe
        DF = pd.DataFrame(base)
        # Converte String para NaN
        DF[DF.columns[1:len(DF.columns)]] = DF[DF.columns[1:len(DF.columns)]].apply(pd.to_numeric, errors='coerce')
        # Apaga as linhas com valores NaN
        DF = DF.dropna()
        # Reseta os indices do novo dataframe
        DF.reset_index(drop=True, inplace=True)
        return DF

    def quartiles(self, base):
        DF = pd.DataFrame(base)
        columns = list(DF)

        # OBS: FALTANTE CHECAR VALORES NEGATIVOS

        Q1 = DF.quantile(0.25, axis=0, numeric_only=True, interpolation='linear')
        Q3 = DF.quantile(0.75, axis=0, numeric_only=True, interpolation='linear')
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR

        return lim_sup, lim_inf

    def clusters(self, n_clusters, base):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)

        kmeans.fit(base)
        base['K classes'] = kmeans.labels_
        base.reset_index(drop=True, inplace=True)

        return base


class Manipulador():
    def __init__(self):

        self.Stack: Gtk.Stack = Builder.get_object("stack")

        self.pasta: Gtk.FileChooserDialog = Builder.get_object('local_base')

        self.modelo_armazenamento: Gtk.ListStore = Builder.get_object("liststore1")
        self.lista_entradas: Gtk.ListStore = Builder.get_object("liststore2")
        self.lista_saidas: Gtk.ListStore = Builder.get_object("liststore3")
        self.lista_dt: Gtk.ListStore = Builder.get_object("liststore5")
        self.escolha_outputs: Gtk.ListStore = Builder.get_object("liststore6")
        self.epocas: Gtk.Entry = Builder.get_object("rna_epochs")
        self.BS: Gtk.Entry = Builder.get_object("rna_BS")
        self.layer1: Gtk.Entry = Builder.get_object("rna_1cam")
        self.layer2: Gtk.Entry = Builder.get_object("rna_2cam")
        self.N_cluster: Gtk.Entry = Builder.get_object("n_clusters")
        self.max_depth: Gtk.Entry = Builder.get_object("max_dep")
        self.N_estimator: Gtk.Entry = Builder.get_object("n_estimators")

        self.combo_box: Gtk.ComboBox = Builder.get_object("combo_box")

        self.coluna_LI: Gtk.TreeViewColumn = Builder.get_object("lim_inf")
        self.coluna_LS: Gtk.TreeViewColumn = Builder.get_object("lim_sup")

        self.entradas = []
        self.saidas = []
        self.maximas = []
        self.minimas = []
        self.entradas_label = []
        self.saidas_label = []

    def on_button_login_clicked(self, button):
        email = Builder.get_object("email").get_text()
        senha = Builder.get_object("senha").get_text()
        lembrar = Builder.get_object("lembrar").get_active()
        self.login(email, senha, lembrar)

    def on_main_window_destroy(self, window):
        Gtk.main_quit()

    def login(self, email, senha, lembrar):
        #if email == 'a' and senha == 'a':
        #    self.mensagem('Bem vindo', 'Usuario Logado com Sucesso', 'emblem-default')
        #    self.Stack.set_visible_child_name("view_inicial")
        #    Window.props.icon_name = 'avatar-default'
        #else:
        #    self.mensagem('Aviso', 'E-mail ou senha incorretos', 'dialog-error')
        self.Stack.set_visible_child_name("view_inicial")

    def on_seleciona_base_clicked(self, button):
        self.pasta.show_all()
        response = self.pasta.run()
        if response == Gtk.ResponseType.OK:
            print("File Selected" + self.pasta.get_filename())
        elif response == Gtk.ResponseType.CANCEL:
            print('Cancelado')

    def on_button_selecionar_clicked(self, button):
        self.pasta.hide()
        self.Stack.set_visible_child_name('view_inicial')
        self.endereco = Builder.get_object('base_address')
        self.arquivo = self.pasta.get_filename()
        self.endereco.set_text(self.arquivo)

    def on_confirmar_clicked(self, button):
        self.arquivo = "C:\msys64\home\luist\Interface-Python\Teste.csv"
        self.base = pd.read_csv(self.arquivo, sep=';', engine='python', decimal=",")
        aux = self.base.columns.values
        aux = aux.reshape(len(aux), 1)
        self.Stack.set_visible_child_name('view_variaveis')
        for row in aux:
            self.modelo_armazenamento.append((row[0], False, False, " ", " "))

    def on_Input_toggled(self, widget, path):
        self.modelo_armazenamento[path][1] = not self.modelo_armazenamento[path][1]

    def on_Output_toggled(self, widget, path):
        self.modelo_armazenamento[path][2] = not self.modelo_armazenamento[path][2]

    def on_aplicar_clicked(self, button):
        self.entradas.clear()
        self.saidas.clear()

        self.filtro = filtros()
        nao_num = self.filtro.funcoes['nao_numerico'](self.base)
        lim_sup, lim_inf = self.filtro.funcoes['quartiles'](nao_num)

        for row in self.modelo_armazenamento:
            self.entradas.append(row[1])
            self.saidas.append(row[2])

        for i in range(len(self.modelo_armazenamento)):
            if self.entradas[i] == True or self.saidas[i] == True:
                # TRATAR ERRO QUARTILES QUANDO VALORES DE STRING
                self.modelo_armazenamento[i][3] = str(round(lim_inf[i - 1],2))
                self.modelo_armazenamento[i][4] = str(round(lim_sup[i - 1],2))

    def on_lim_sup_edited(self, widget, path, text):
        self.modelo_armazenamento[path][4] = float(text)

    def on_lim_inf_edited(self, widget, path, text):
        self.modelo_armazenamento[path][3] = float(text)

    def on_button_cancelar_clicked(self, button):
        self.pasta.hide()
        self.Stack.set_visible_child_name('view_inicial')

    def on_avancar_clicked(self, button):
        self.Stack.set_visible_child_name('view_base')
        for i in range(len(self.modelo_armazenamento)):
            if self.entradas[i] == True:
                self.entradas_label.append(self.modelo_armazenamento[i][0])
                self.lista_entradas.append([self.modelo_armazenamento[i][0]])
                self.minimas.append(float(self.modelo_armazenamento[i][3]))
                self.maximas.append(float(self.modelo_armazenamento[i][4]))

            if self.saidas[i] == True:
                self.saidas_label.append(self.modelo_armazenamento[i][0])
                self.lista_saidas.append([self.modelo_armazenamento[i][0]])
                self.minimas.append(float(self.modelo_armazenamento[i][3]))
                self.maximas.append(float(self.modelo_armazenamento[i][4]))
                self.escolha_outputs.append([self.modelo_armazenamento[i][0]])

        aux = np.logical_or(self.entradas, self.saidas)
        self.maximas = np.asarray(self.maximas).reshape(1, len(self.maximas))
        self.minimas = np.asarray(self.minimas).reshape(1, len(self.minimas))

        self.base_aux = self.base.drop(self.base.iloc[:, ~aux], axis=1)
        self.base_aux = self.base_aux.where(self.base_aux < self.maximas)
        self.base_aux = self.base_aux.where(self.base_aux > self.minimas)
        self.base_aux.dropna(inplace=True)

        self.n_cl = int(self.N_cluster.get_text())
        self.base_aux = self.filtro.funcoes['clusterizacao'](self.n_cl, self.base_aux)

    def on_treinar_rna_clicked(self, button):
        self.Stack.set_visible_child_name('view_rna')

    def on_button_treinar_RNA_clicked(self, button):
        self.alg_IA = algoritmos_AI()

        epocas = int(self.epocas.get_text())
        BS = int(self.BS.get_text())
        layer1 = int(self.layer1.get_text())
        layer2 = int(self.layer2.get_text())

        self.teste = self.alg_IA.tipos['RNA_2C'](self.base_aux, layer1, layer2, epocas, BS,
                                                 self.n_cl, self.entradas_label, self.saidas_label)

    def on_treinar_dt_clicked(self, button):
        self.Stack.set_visible_child_name('view_dt')

    def on_train_tree_clicked(self, button):
        self.alg_IA = algoritmos_AI()

        max_dep = int(self.max_depth.get_text())
        n_estim = int(self.N_estimator.get_text())

        [self.dados_dt, self.previsoes_dt, self.resultados_dt] = self.alg_IA.tipos['Dec_Tree'](self.base_aux, max_dep, n_estim, self.n_cl,
                                                   self.entradas_label, self.saidas_label)

        self.resultados_dt = pd.DataFrame(self.resultados_dt)

        i = 0
        while i < len(self.saidas_label):
            self.lista_dt.append((self.saidas_label[i],str(round(np.mean(np.asarray(self.resultados_dt[:][i])),2))))
            print(np.mean(np.asarray(self.resultados_dt[:][i])))
            i+=1

    def on_estimar_clicked(self, button):
        self.Stack.set_visible_child_name('view_estima')

    def on_escolhe_in_out_clicked(self, button):
        self.Stack.set_visible_child_name('view_variaveis')

    def on_voltar_clicked(self, button):
        self.Stack.set_visible_child_name('view_base')

    def on_botao_base_clicked(self, button):
        self.modelo_armazenamento.clear()
        self.Stack.set_visible_child_name('view_inicial')

    def mensagem(self, param, param1, param2):
        mensagem: Gtk.MessageDialog = Builder.get_object("mensagem")
        mensagem.props.text = param
        mensagem.props.secondary_text = param1
        mensagem.props.icon_name = param2
        mensagem.show_all()
        mensagem.run()
        mensagem.hide()

    def on_mostrar_header_clicked(self, button):
        self.modelo_armazenamento.clear()
        self.modelo_armazenamento.append(('teste'))

    def on_analise_grafica_clicked(self,button):

        grafico = Builder.get_object('graf')
        janela_grafico = Builder.get_object('grafico')

        fig, ax = plt.subplots()
        ax.set_title('Dados Teste vs Dados Previstos')
        ax.set(xlabel='Dados', ylabel='Output Escolhida')
        amostras = len(self.previsoes_dt[0][0])
        X = range(amostras)

        ax.plot(X, self.dados_dt[0][1].loc[:, self.lista_saidas[self.combo_box.get_active()][0]], color="crimson", label="Dados Reais", linewidth=0.5)
        ax.plot(X,self.previsoes_dt[0][self.combo_box.get_active()] , color="dodgerblue",
                 alpha = 0.9, c ="red", linestyle='dashed', label="Dados do Modelo", linewidth = 0.5)
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvas(fig)

        grafico.add_with_viewport(canvas)
        janela_grafico.show_all()

    def on_combo_box_changed(self,combo):
        print(self.lista_saidas[self.combo_box.get_active()][0])



Builder = Gtk.Builder()
Builder.add_from_file("user_interface.glade")
Builder.connect_signals(Manipulador())
Window: Gtk.Window = Builder.get_object("main_window")
Window.show_all()
Gtk.main()
