import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk
import pandas as pd


class user():
    def __init__(self, var):
        self.Variaveis = var


class Manipulador():
    def __init__(self):
        self.modelo_armazenamento: Gtk.ListStore = Builder.get_object("liststore1")
        self.Stack: Gtk.Stack = Builder.get_object("stack")
        self.pasta: Gtk.FileChooserDialog = Builder.get_object('local_base')
        self.banco_dados = []
        self.model = Gtk.ListStore(str)
        self.liststore = Gtk.ListStore(str, str)



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
            print('Cancel Clicked')

    def on_button_selecionar_clicked(self, button):
        self.pasta.hide()
        self.Stack.set_visible_child_name('view_inicial')
        self.endereco = Builder.get_object('base_address')
        self.arquivo = self.pasta.get_filename()
        self.endereco.set_text(self.arquivo)

    def on_confirmar_clicked(self, button):
        base = pd.read_csv(self.arquivo, sep=';', engine='python', decimal=",")
        aux = base.columns.values
        aux = aux.reshape(len(aux), 1)
        self.model.append(aux)
        print(aux.reshape(len(aux), 1))
        self.banco_dados.append(aux)
        print(self.banco_dados)
        #self.Stack.set_visible_child_name('view_variaveis')
        self.liststore.append(aux)


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
