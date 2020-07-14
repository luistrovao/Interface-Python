import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
import pandas as pd



class filtros():
    def __init__(self):
        self.funcoes = {
            "nao_numerico": self.nao_num,
            "quartiles": self.quartiles,
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



class Manipulador():
    def __init__(self):
        self.modelo_armazenamento: Gtk.ListStore = Builder.get_object("liststore1")
        self.coluna_LI: Gtk.TreeViewColumn = Builder.get_object("lim_inf")
        self.coluna_LS: Gtk.TreeViewColumn = Builder.get_object("lim_sup")
        self.Stack: Gtk.Stack = Builder.get_object("stack")

        self.pasta: Gtk.FileChooserDialog = Builder.get_object('local_base')
        self.entradas = []
        self.saidas = []

    def on_button_login_clicked(self, button):
        email = Builder.get_object("email").get_text()
        senha = Builder.get_object("senha").get_text()
        lembrar = Builder.get_object("lembrar").get_active()
        self.login(email, senha, lembrar)

    def on_main_window_destroy(self, window):
        Gtk.main_quit()

    def login(self, email, senha, lembrar):
        if email == 'a' and senha == 'a':
            self.mensagem('Bem vindo', 'Usuario Logado com Sucesso', 'emblem-default')
            self.Stack.set_visible_child_name("view_inicial")
            Window.props.icon_name = 'avatar-default'
        else:
            self.mensagem('Aviso', 'E-mail ou senha incorretos', 'dialog-error')

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

        self.base = pd.read_csv(self.arquivo, sep=';', engine='python', decimal=",")
        aux = self.base.columns.values
        aux = aux.reshape(len(aux), 1)
        self.Stack.set_visible_child_name('view_variaveis')

        for row in aux:
            self.treeiter = self.modelo_armazenamento.append((str(row), False, False, 0, 0))

    def on_Input_toggled(self, widget, path):
        self.modelo_armazenamento[path][1] = not self.modelo_armazenamento[path][1]

    def on_Output_toggled(self, widget, path):
        self.modelo_armazenamento[path][2] = not self.modelo_armazenamento[path][2]

    def on_aplicar_clicked(self, button):
        self.entradas.clear()
        self.saidas.clear()

        filtro = filtros()
        TESTE = filtro.funcoes['nao_numerico'](self.base)
        lim_sup,lim_inf = filtro.funcoes['quartiles'](TESTE)
        print(lim_inf[0])

        for row in self.modelo_armazenamento:
            self.entradas.append(row[1])
            self.saidas.append(row[2])

        for i in range(len(self.modelo_armazenamento)):
            if self.entradas[i] == True or self.saidas[i] == True:
                self.modelo_armazenamento[i][3] = lim_inf[i-1]
                self.modelo_armazenamento[i][4] = lim_sup[i-1]






    def on_button_cancelar_clicked(self, button):
        self.pasta.hide()
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


Builder = Gtk.Builder()
Builder.add_from_file("user_interface.glade")
Builder.connect_signals(Manipulador())
Window: Gtk.Window = Builder.get_object("main_window")
Window.show_all()
Gtk.main()
