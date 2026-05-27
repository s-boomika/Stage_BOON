# Graphical interface for BooN design and analysis
# Author: Franck Delaplace
# Creation date: February 2024
# Co-author: Boomika SELVARADJOU
# Modification date: April 2026

#In comments:
#DEF means definition which is a code part gathering functions related to a process or an object definition,
#STEP means main steps
#WARNING is a warning.
#These terms can be used to color comments in PyCharm or else.

# Standard library imports
import sys
import os
import math
import re
import numpy as np

# Third-party imports: import functions only used 
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

#from netgraph import EditableGraph #not use editgraph
from sympy import SOPform, symbols
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import is_cnf, is_dnf, is_nnf
from sympy.logic.boolalg import And, Not
from sympy.parsing.sympy_parser import parse_expr
from pulp import PULP_CBC_CMD

#PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QToolButton, QMenu, QWidget, QGridLayout, QPushButton, QWidgetAction, QMessageBox, QColorDialog
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem, QColor, QCursor
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.uic import loadUi

from turtle import color, pos
from matplotlib import colors
from matplotlib.pylab import norm
from matplotlib.patches import Rectangle
from tabulate import tabulate

#Local 
import boon
from boon import BooN, SIGNCOLOR, COLORSIGN, EXTSBML, EXTXT, BOONSEP, PYTHONSKIP, PYTHONHEADER
import boon.logic as logic
from boon.logic import LOGICAL, SYMPY, MATHEMATICA, JAVA, BOOLNET

import BooNGui.booneries_rc  #resources

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

mpl.use("Qt5Agg")

#Parameters
HSIZE: int = 10  #size of the history
STYLE: dict = {"Logical": LOGICAL, "Java": JAVA, "Python": SYMPY, "Mathematica": MATHEMATICA, "BoolNet": BOOLNET}
ICON01: dict = {None: ":/icon/resources/none.svg", True: ":/icon/resources/true.svg", False: ":/icon/resources/false.svg"}  #True/False icons
MODELBOUND: int = 8  #Size bound of the dynamics model in terms of variables.
LPSOLVER = PULP_CBC_CMD  #LP-Solver related to the controllability resolution. (destify)

#Integer regular expression
INTPAT: str = r"\s*-?[0-9]+\s*"
# basic family color (white)
BASIC_FAMILY_COLOR : tuple[int, int, int ] = (1 ,1 ,1 ) #white

class Boonify(QMainWindow):
    """
    Represents the main application window for managing and designing BooNs (Boolean Networks).
    Provides a GUI with a comprehensive set of functionalities. This class is primarily responsible for initializing
    and connecting GUI widgets, setting up callbacks, and constructing an editable graph
    for network interaction design.
        :ivar boon: The current Boolean Network object being managed or displayed.
        :type boon: BooN
        :ivar filename: The name of the file associated with the current BooN (empty if no file is loaded or saved).
        :type filename: str
        :ivar history: Maintains a history list of BooN objects for undo/redo functionalities.
        :type history: list
        :ivar hindex: The index of the last BooN added to the history.
        :type hindex: int
        :ivar hupdate: Flag indicating whether the history should be updated.
        :type hupdate: bool
        :ivar saved: Flag indicating if the current BooN has been saved.
        :type saved: bool
        :ivar QView: Widget for displaying BooN visualization.
        :type QView: QWidget | None
        :ivar QStableStates: Widget for displaying stable states of BooNs.
        :type QStableStates: QWidget | None
        :ivar QModel: Widget for displaying the dynamic model of the BooN.
        :type QModel: QWidget | None
        :ivar QControllability: Widget for displaying BooN controllability analysis.
        :type QControllability: QWidget | None
        :ivar editgraph: Editable graphical representation of the BooN's interaction graph.
        :type editgraph: EditableGraph | None
        :ivar disablecallback: Flag indicating whether design callbacks are disabled.
        :type disablecallback: bool
        :ivar designsize: Scaling factor related to graphics elements in the EditableGraph.
        :type designsize: float
        :ivar worker: Background thread for processing long-running BooN operations.
        :type worker: Threader
        :ivar canvas: Matplotlib canvas for rendering the network design figure in the GUI.
        :type canvas: FigureCanvas
    """
    #Constructor
    def __init__(self):
        super(Boonify, self).__init__()
        #loadUi('BooNGui/boonify.ui', self)
        ui_path = os.path.join(os.path.dirname(__file__), 'BooNGui', 'boonify.ui')
        loadUi(ui_path, self)
        self.setGeometry(600, 100, 800, 800)

        #STEP: initialize the Gui state
        self.boon = BooN()  #Current BooN
        self.network = Network()
        self.filename = ""  #Current filename
        self.history = [None] * HSIZE  #History
        self.hindex = 0  #Index of the last BooN added in the history.
        self.hupdate = False  #Flag determining whether the history is updated.
        self.saved = True  #Flag determining whether the current BooN is saved.

        self.QView = None  #Widget of the View
        self.QStableStates = None  #Widget of the stable states
        self.QModel = None  #Widget of the dynamics model
        self.QControllability = None  #Widget of the controllability

        self.disablecallback = True  #Flag indicating whether the BooN design callback function is enabled, initially disabled (True) because the editgraph is not set up.
        self.designsize = 2.  #Size related to the EditableGraph and used for many parameters. It is modified when the window is rescaled to let the nodes and edges width invariant.
        self.zoom_factor = 1.0
        self.edge_source = None
        self.edge_family_colors = {}
        self.default_edge_sign = 1   # 1 = green, -1 = red
        self.edge_preview_target = None

        self.setup_callbacks()

        self.display_saved_flag()  #Show the save flag in the status bar

        #STEP: Connect callback functions to Menu
        #File Management
        self.ActionOpen.triggered.connect(self.open)
        self.ActionSave.triggered.connect(self.save)
        self.ActionSaveAs.triggered.connect(self.saveas)
        self.ActionImport.triggered.connect(self.importation)
        self.ActionExport.triggered.connect(self.exportation)
        self.ActionQuit.triggered.connect(self.quit)

        #Help
        self.ActionHelp.triggered.connect(self.help)
        self.ActionUndo.triggered.connect(self.undo)
        self.ActionRedo.triggered.connect(self.redo)

        #Network Management
        self.ActionView.triggered.connect(self.view)
        self.ActionModel.triggered.connect(self.model)
        self.ActionStableStates.triggered.connect(self.stablestates)
        self.ActionControllability.triggered.connect(self.controllability)

        #STEP: Initialization of a thread for long run application --> controllability
        #The thread is created and started.
        self.worker = Threader()

        #STEP: Initialize the Matplotlib Canvas for network design
        self.network = Network()
        fig = plt.figure()  #plt.figure is necessary to create a manager
        manager = fig.canvas.manager
        self.canvas = FigureCanvas(fig)
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.figure.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust the window s.t. The network fully fills it.
        self.canvas.figure.canvas.manager = manager  #Assign the manager in the canvas to be accessible by EditGraph.

        # STEP: Create graph editor ON THE REAL CANVAS
        self.graph_editor = Graph(self.canvas)

        self.graph_editor.graph_changed.connect(self.on_graph_changed)

        # STEP: Insert canvas into GUI
        self.DesignCanvas.addWidget(self.canvas)

        # STEP: Setup graph
        self.graph_editor.setup_design(self.boon)

        self.disablecallback = False

        #STEP: Enable key_press_event:
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()
        
        #STEP: Connect mouse and keyboard events
        self.canvas.mpl_connect('button_press_event', self.graph_editor.on_canvas_press)
        self.canvas.mpl_connect('button_release_event', self.graph_editor.on_canvas_release)
        self.canvas.mpl_connect('motion_notify_event', self.graph_editor.on_canvas_motion)
        self.canvas.mpl_connect('key_press_event', self.graph_editor.on_key_press)

        #STEP: Graph actions
        self.actionRenameNode.triggered.connect(self.graph_editor.rename_node)
        self.actionDelete.triggered.connect(self.graph_editor.delete_selection)
        self.actionZoomIn.triggered.connect(self.graph_editor.zoom_in)
        self.actionZoomOut.triggered.connect(self.graph_editor.zoom_out)

        #STEP: Edge Sign actions
        self.actionEdgePlus.triggered.connect(lambda: self.graph_editor.set_default_edge_sign(1))
        self.actionEdgeMinus.triggered.connect(lambda: self.graph_editor.set_default_edge_sign(-1))

        #STEP: Setup color menu
        self.graph_editor.setup_color_menu(self)

        # Connect action directly to opening palette
        self.actionFamilyColor.triggered.connect(self.graph_editor.open_color_palette)
        self.actionHideFamilyColor.triggered.connect(self.graph_editor.toggle_family_colors)
        # setup palette (no QAction binding here anymore)
        self.graph_editor.setup_color_menu(self)


    def on_graph_changed(self):
        """
        Synchronize Graph -> BooN when the graph editor changes.
        """
        self.boon = self.network.graph_to_boon(
            self.graph_editor,
            current_boon=self.boon
        )
        self.add_history()
        self.refresh()

    def setup_callbacks(self):
        """
        Connect UI signals to controller callbacks.
            Links the graph editor's `graph_changed` signal to the handler
            `on_graph_changed`, if the graph editor exists.
        """
        if hasattr(self, "graph_editor"):
            self.graph_editor.graph_changed.connect(self.on_graph_changed)

    #DEF: FILE MANAGEMENT
    def open(self):
        """
        Open a file using a file dialog, load its contents, and update the application state.
        This method presents a file dialog to the user to choose a file with the specific extension.
        After successfully selecting a file, the method loads its content,
        refreshes the application view, initializes the design setup, and clears the history.
            :return: None
        """
        filename = QFileDialog.getOpenFileName(self, "Open file", "", "Boon Files (*.boon);; All Files (*);;")
        if filename:
            self.filename = filename[0]
            self.boon = BooN.load(filename[0])
            self.refresh()
            self.graph_editor.setup_design(self.boon)
            self.history_raz()  #clear history

    def save(self):
        """
        Saves the current state to a file.
        This method attempts to save the current state to a file using the
        filename attribute. If a filename is already provided, the current
        state is saved to this file, and the saved status flag is updated.
        If no filename is specified, the method triggers a save-as operation
        to determine the filename.
            :return: None
        """
        if self.filename:
            self.boon.save(self.filename)
            self.display_saved_flag()  #update the saved flag
        else:
            self.saveas()

    def saveas(self):
        """
        Saves the current content to a file. The method opens a "Save As" dialog
        to specify a filename and filepath, saves the content to the chosen file,
        and updates relevant components accordingly.
        """
        filename = QFileDialog.getSaveFileName(self, "Save", "", "Boon Files (*.boon);; All Files (*);;")
        if filename:
            self.filename = filename[0]
            self.boon.save(self.filename)
            self.display_saved_flag()  #update the saved flag

    def importation(self):
        """
        Handles the importation of a file and initializes the respective properties
        based on the detected file format. This function utilizes specific handlers
        to import files in supported formats such as BoolNet (.bnet), Python (.txt),
        and SBML (.sbml or .xml). It ensures that the application state is updated
        after the import, refreshing the GUI, setting up the design, and clearing
        the history if the file is successfully imported.
        If an unsupported file extension is provided, it displays a critical error
        message to inform the user about acceptable formats.
            :return: None
        """
        filename = QFileDialog.getOpenFileName(self, "Import from files", "", "Text or SBML Files (*.bnet *.txt *.xml *.sbml);; All Files (*);;")
        if filename:
            self.filename = None  #no file name since the BooN is not saved in the internal format.
            _, extension = os.path.splitext(filename[0])
            match extension:  #Select the appropriate format to import a file
                case ".bnet":  #BoolNet Format
                    self.boon = BooN.from_textfile(filename[0])
                case ".txt":  #Python Format
                    self.boon = BooN.from_textfile(filename[0], sep=BOONSEP, assign='=', ops=SYMPY, skipline=PYTHONSKIP)
                case ".sbml":  #SBML Format
                    self.boon = BooN.from_sbmlfile(filename[0])
                case ".xml":  #SBML Format
                    self.boon = BooN.from_sbmlfile(filename[0])
                case _:
                    QMessageBox.critical(self, "File extension error", f"The extension is unknown. \nFound {extension}\nAdmitted extension: .txt, .bnet, .sbml, .xml")
            self.refresh()
            self.graph_editor.setup_design(self.boon)
            self.history_raz()  #clear history

    def exportation(self):
        """
        Exports data to a specified file format (BoolNet or Txt).
        The method allows exporting data to a file in the BoolNet or Txt format based
        on the provided file name and its extension.
        The BoolNet format is supported with a `.bnet` extension, and the Python-compatible
        format is supported with a `.txt` extension.
            :raises ValueError: If the filename extension is unsupported or invalid.
            :return: None
        """
        filename = QFileDialog.getSaveFileName(self, "Export to BoolNet or Python format.", "", "Text Files (*.bnet *.txt);;")
        if filename:
            _, extension = os.path.splitext(filename[0])
            match extension:
                case ".bnet":
                    self.boon.to_textfile(filename[0])
                case ".txt":
                    self.boon.to_textfile(filename[0], sep=BOONSEP, assign='=', ops=SYMPY, header=PYTHONHEADER)

    def quit(self):
        """
        Terminates the application based on user confirmation and save status.
        This method checks if the BooN has been saved. If it is not saved,
        it prompts the user with a dialog to choose whether to save, cancel, or quit without saving.
            :return: None
        """
        #check if the BooN is saved
        if self.saved:
            quitting = True
        else:  #Otherwise, ask whether it must be saved.
            reply = QMessageBox.question(
                self,
                "Quit",
                "Are you sure you want to quit? \nThe BooN is not saved.",
                QMessageBox.Save | QMessageBox.Close | QMessageBox.Cancel,
                QMessageBox.Save)
            match reply:
                case QMessageBox.Save:
                    self.save()
                    quitting = True
                case QMessageBox.Close:
                    quitting = True
                case QMessageBox.Cancel:
                    quitting = False
                case _:
                    quitting = False
        if quitting:
            app.quit()

    #noinspection PyMethodOverriding
    def closeEvent(self, event):
        """
        Handles the close event for the application window.
        This method ensures that when the application's close event is triggered, it invokes the quit method
        to handle quitting properly. The `event.ignore()` ensures that the close event is ignored unless
        the application is closed successfully prior to that, in which case `event.ignore()` will not execute.
            :param event: The close event being processed, which carries information about the window-closing action.
            :type event: QCloseEvent
            :return: None
        """
        self.quit()
        event.ignore()  #WARNING: If the application is closed when quit() is triggered, this line will not be executed.

    #DEF: HISTORY MANAGEMENT
    def history_raz(self):
        """
        Resets the history and initializes it with the current state of BooN. It clears
        the history, sets the history index to the starting position, and ensures that
        the history update flag is reset.
            :ivar history: A list to record the history of BooN states. Initially cleared
                and reinitialized to a default size, HSIZE.
            :ivar hindex: An integer representing the index position of the last BooN state
                added to the history. Reset to 0 after cleaning history.
            :ivar hupdate: A boolean flag indicating whether the history is updated. Set to
                False after resetting the history.
        """
        self.history = [None] * HSIZE  #History cleaning
        self.hindex = 0  #Index of the last BooN added in the history
        self.add_history()  #Initialize the history with the current BooN
        self.hupdate = False  #Flag determining whether the history is updated
        self.display_saved_flag()  #The BooN is saved.

    def undo(self):
        """
        Restores the previous state from the history while preventing disruptive updates.
        This method decrements the history index and restores the previous state of the
        `boon` attribute using the corresponding entry in the `history`. It ensures
        that no unwanted side effects occur during the update process. After restoring
        the state, it refreshes the current state and reinitializes necessary design
        parameters.
            :raises IndexError: Raised if the history operation fails due to an 
                invalid index or empty history.
            :return: None
        """
        hindex = (self.hindex - 1) % HSIZE
        if self.history[hindex]:
            self.disablecallback = True  #Prevent disruptive updates by disabling callback.
            self.boon = self.history[hindex].copy()  #restore the previous BooN
            self.hindex = hindex  #update the history index
            self.refresh()
            self.graph_editor.setup_design(self.boon)
            self.graph_editor.refresh_next_node_id()

            self.disablecallback = False

    def redo(self):
        """
        Redo the last undone operation by navigating through the history list and updating
        the current state. This method ensures that callback functions are temporarily
        disabled during the operation to avoid disruptive updates.
        """
        hindex = (self.hindex + 1) % HSIZE
        if self.history[hindex]:
            self.disablecallback = True  #Prevent disruptive updates by disabling callback.
            self.boon = self.history[hindex].copy()  #copy the current Boon
            self.hindex = hindex
            self.refresh()
            self.graph_editor.setup_design(self.boon)
            self.graph_editor.refresh_next_node_id()

            self.disablecallback = False

    def add_history(self):
        """
        Adds the current BooN to the history if it differs from the last saved state. This method ensures
        that changes to the BooN descriptor are tracked and recorded appropriately in the history buffer.
        It also manages internal state flags to enable or disable disruptive behaviors such as callbacks
        when updating history. The history index is circularly updated to maintain a fixed-size
        history buffer.
            attribute self.hindex: Index of the current position in the history.
            attribute self.boon.desc: Descriptor of the current BooN.
            attribute self.disablecallback: Boolean that disables callback updates when `True`.
            attribute self.hupdate: Boolean that indicates whether the history has been updated.
            :return: None
        """
        hindex = self.hindex
        if self.boon != self.history[hindex]:  #WARNING: The BooN comparison operates on descriptors only, not on positions or style.

            self.disablecallback = True  #Prevent disruptive updates by disabling callback.
            self.hupdate = True  #Descriptor is changed.

            hindex = (hindex + 1) % HSIZE  #Update the history
            self.history[hindex] = self.boon.copy()
            self.hindex = hindex

            if self.history[hindex] and self.history[hindex].desc:  #The current BooN is modified
                self.display_saved_flag(False)
            self.disablecallback = False  #Enable the design callback
        else:
            self.hupdate = False  #No changes.

    def show_history(self):
        """
        Displays the entire history of changes, showing each entry formatted according
        to its identifier and rendering logic using a tabulated structure. The current
        history index is highlighted, and associated data details are presented using
        a plain tabular format. If a history entry lacks data, it displays a placeholder.
        """
        view = [([i] if i == self.hindex else i,
                 tabulate([(var, logic.prettyform(eq, theboon.style)) for var, eq in theboon.desc.items()], tablefmt='plain')
                 if theboon else '-') for i, theboon in enumerate(self.history)]

        os.system('cls')
        print(tabulate(view, tablefmt='grid'))

    def refresh(self):
        """
        Refreshes all visible components, such as BooN View, Stable States View,
        Model View, and Controllability View. Each visible component is reinitialized
        or updated via its relevant function. This ensures that all active components
        reflect the most current state.
            :return: None
        """
        if self.QView and self.QView.isVisible():  #Refresh the BooN View if opened.
            self.QView.initialize_view()
        if self.QStableStates and self.QStableStates.isVisible():  #Refresh the stable states View if opened.
            self.QStableStates.stablestates()
        if self.QModel and self.QModel.isVisible():  #Refresh the Model view if opened.
            self.QModel.modeling()
        if self.QControllability and self.QControllability.isVisible():  #Refresh the Controllability View if opened.
            self.QControllability.initialize_controllability()

    #DEF: WIDGETS OPENING
    def help(self):
        """
        Provides functionality to call and display help using an external Help object.
            :return: None
        """
        thehelp = Help(self)
        thehelp.show()

    def view(self):
        """
        Initializes and displays the View object for the class instance.
            :return: None
        """
        self.QView = View(self)
        self.QView.show()

    def stablestates(self):
        """
        Displays stable states.
            :return: None
        """
        self.QStableStates = StableStates(self)
        self.QStableStates.show()

    def model(self):
        """
        Display the dynamical model if the number of variables does not exceed the defined model boundary
            :return: None
        """
        if len(self.boon.variables) > MODELBOUND:
            QMessageBox.critical(self, "No Model", f"The number of variables exceeds {MODELBOUND}.\nThe model cannot be drawn.")
            return
        self.QModel = Model(self)
        self.QModel.show()

    def controllability(self):
        """
        Defines a function that initializes and displays the controllability object
        related to the current self-instance.
        attributes QControllability : Controllability
        And instance of the `Controllability` class associated with the current object.
        It encapsulates functionality to assess or manage controllability features.
            :return: None
        """
        self.QControllability = Controllability(self)
        self.QControllability.show()

    def display_saved_flag(self, val: bool = True):
        """
        Displays a flag in the status bar indicating whether the data has been saved.
        This method updates the status bar message based on the save state. If the
        data is saved, a large empty circle is displayed. If the data is not saved,
        a large black circle is displayed.
            :param val: They save a status flag. If True, the data is saved, and the large
                empty circle is shown. If False, the data is not saved, and the
                large black circle is displayed.
            :type val: bool
        """
        NOTSAVED: str = '\u2B24'  #Large black circle
        SAVED: str = '\u25CB'  #Large empty circle
        self.saved = val
        if self.saved:
            self.statusBar().showMessage(SAVED)
        else:
            self.statusBar().showMessage(NOTSAVED)


class Graph(QObject):

    graph_changed = pyqtSignal()

    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.axes = canvas.axes
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.node_labels = {}
        self.next_node_id = 1
        self.edge_colors = {}
        self.edge_labels = {}
        self.edge_modules = {}
        self.default_edge_sign = 1   # + edge = green by default
        self.selected_nodes = set()
        self.selected_edge = None
        self.zoom_factor = 1.0
        self.edge_family_colors = {}
        self.show_family_colors = True

        self.double_click_node = None
        self.edge_preview = None
        self.edge_source = None
        self.tracked_positions = {}
        self.drag_start_pos = None
        self.selection_rect_start = None
        self.selection_rect_patch = None
        self.dragging_nodes = False
        self.is_dragging = False

    """
    Model / Logic
    """
    #DEF: NETWORK DESIGN
    def setup_design(self, boon):
        """
        Sets up the visual and logical design of the interaction graph.

        This method initializes the graph structure from the provided BooN model,
        configures node positions, labels, and edge attributes, and prepares the
        canvas for rendering. It also synchronizes edge signs and initializes
        visual styling used in the interactive editor.

            :param boon: External model containing the interaction graph and optional
                node positions.
            :type boon: BooN

            :ivar SIGNCOLOR: Mapping from edge sign values to display colors.
            :type SIGNCOLOR: dict
        """
        self.boon = boon
        self.canvas.axes.clear()
        self.SIGNCOLOR = SIGNCOLOR
        #STEP: Creation of the pattern-network used for edge styling.
        #To prevent its inclusion in the BooN, the variable names are strings while the other nodes are integers or symbols.
        #STEP: Add  the interaction graph of the current BooN.
        ig = boon.interaction_graph  #Get the IG.
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(ig.nodes())
        for u, v, data in ig.edges(data=True):
            self.graph.add_edge(u, v, sign=data.get("sign", 1))

        self.node_positions = getattr(boon, "pos", None) or nx.spring_layout(self.graph)
        self.node_labels = {node: str(node) for node in self.graph.nodes()}
        
        self.edge_colors = {}
        for edge in self.graph.edges():
            sign = ig.get_edge_data(*edge).get("sign", 1)
            # store sign INSIDE editable graph too
            self.graph[edge[0]][edge[1]]["sign"] = sign
            self.edge_colors[edge] = self.SIGNCOLOR[sign]

        self.redraw_graph()

        existing_ids = [
            n for n in self.graph.nodes()
            if isinstance(n, int)
            ]
        if existing_ids:
            self.next_node_id = max(existing_ids) + 1
        else:
            self.next_node_id = 1

    def _track_node_positions(self, dx, dy):
        """
        Tracks node position updates during dragging without committing changes.

        Updates the internal node position dictionary for all selected nodes
        based on the drag delta and refreshes the visualization.

            :param dx: Horizontal movement delta.
            :type dx: float

            :param dy: Vertical movement delta.
            :type dy: float
        """
        #Update tracked positions
        for node in self.selected_nodes:
            if node in self.node_positions:
                x, y = self.node_positions[node]
                self.node_positions[node] = (x + dx, y + dy)

        self.redraw_graph()

    def _commit_tracked_positions(self):
        """
        Commits dragged node positions to the underlying BooN model.

        Updates the persistent position storage for all selected nodes and
        triggers a redraw of the graph.

            :raises AttributeError: If the BooN model has no position attribute.
        """
        # Update boon positions
        if hasattr(self.boon, 'pos') and hasattr(self.boon, "pos"):
            for node in self.selected_nodes:
                if node in self.node_positions:
                    self.boon.pos[node] = self.node_positions[node]
       
        self.redraw_graph()
        print(f"Committed positions for nodes: {self.selected_nodes}")

    def _apply_edge_color(self, rgb):
        """
        Applies a visual color to the currently selected edge.

        Updates the edge color mapping for the selected edge in both directions
        (if bidirectional) and refreshes the graph display.

            :param rgb: RGB color tuple (values in range [0, 1]).
            :type rgb: tuple[float, float, float]
        """
        if not self.selected_edge:
            return
        u, v = self.selected_edge
        #if bidirectional, apply to both directions
        self.edge_colors[(u,v)] = rgb #Update visual color
        self.edge_colors[(v,u)] = rgb
        self.redraw_graph()

    def change_edge_sign(self, sign):
        """
        Changes the sign (and color) of the currently selected edge.

        Updates both the logical graph representation and the visual color mapping.
        Emits a graph update signal after modification.

            :param sign: New sign value for the edge.
            :type sign: int

            :raises QMessageBox.warning: If no edge is currently selected.
        """
        if not self.selected_edge: #if no edge selected, show error
            QMessageBox.warning(None, "No edge selected", "Please click an edge first.")
            return
        
        u, v = self.selected_edge

        # update visual color
        self.edge_colors[(u, v)] = self.SIGNCOLOR[sign]

        # update logical sign
        if self.graph.has_edge(u, v):
            self.graph[u][v]["sign"] = sign

        self.redraw_graph()
        self.graph_changed.emit()

    def set_default_edge_sign(self, sign):
        """
        Sets the default sign used for newly created edges.

            :param sign: Edge sign (typically +1 or -1).
            :type sign: int
        """
        self.default_edge_sign = sign
        print(f"Default edge sign set to {sign}")

    def _add_new_node(self, x, y):
        """
        Add a new node at the specified position.
            :param x: X coordinate
            :param y: Y coordinate
        Find next integer id for the new node
        """
        #Internal graph ID
        new_id = self.next_node_id
        self.next_node_id += 1

        new_label = self.next_default_label()           #Visible label

        self.graph.add_node(new_id)                     #Creaste node
        self.node_positions[new_id]=(x,y)               #Store positions
        self.node_labels[new_id]= new_label             #Store displayed label

        #Redraw
        self.redraw_graph()
        self.graph_changed.emit()

    def refresh_next_node_id(self):
        """
        Recomputes the next available integer node ID.

        Scans existing integer nodes and updates the internal counter to avoid
        ID collisions.
        """
        int_nodes = [n for n in self.graph.nodes if isinstance(n, int)]

        if int_nodes:
            self.next_node_id = max(int_nodes) + 1
        else:
            self.next_node_id = 1
    
    def next_default_label(self):
        """
        Generates the next available default node label.

        Only labels matching the pattern "x<number>" are considered. Custom
        renamed nodes are ignored when computing the next index.
            :return: Next generated label (e.g., "x3").
            :rtype: str

        Example:
            x1, x2 -> x3
            tom, x2, x3 -> x4
        """
        max_index = 0
        for label in self.node_labels.values():
            if not isinstance(label, str):
                continue
            
            match = re.fullmatch(r"x(\d+)", label.strip())
            if match:
                number = int(match.group(1))
                if number > max_index:
                    max_index = number
        return f"x{max_index + 1}"

    def _create_edge(self, source, target):
        """
        Creates a directed edge between two nodes.

        Adds the edge to the internal graph structure, assigns default sign and
        visual styling, clears selection state, and triggers a redraw.

            :param source: Source node identifier.
            :param target: Target node identifier.
            :type source: any
            :type target: any
        """
        # clear preview safely first
        self._clear_edge_preview()

        if source is None or target is None:
            return

        if self.graph.has_edge(source, target):
            print(f"Edge {source} -> {target} already exists")
            return
        self.graph.add_edge(source, target, sign=self.default_edge_sign)
        self.edge_family_colors[(source, target)] = (1.0, 1.0, 1.0)
        self.edge_colors[(source, target)] = (self.SIGNCOLOR[self.default_edge_sign]) 
        self.selected_nodes.clear()
        self.selected_edge = None
        self.double_click_node = None
        print(f"Added edge to graph: {source} -> {target}, self-loops: {[(u,v) for u,v in self.graph.edges() if u == v]}")
        self.redraw_graph()
        self.graph_changed.emit()
        print(f"Edge created: {source} -> {target}")

    def _compute_self_loop_angle(self, node):
        """
        Computes an optimal placement angle for a self-loop edge.

        Determines a direction that avoids overlapping neighboring nodes by
        analyzing angular spacing around the node.

            :param node: Node identifier.
            :type node: any

            :return: Angle in radians for loop placement.
            :rtype: float
        """
        neighbors = set(self.graph.predecessors(node)) | set(self.graph.successors(node))
        neighbors.discard(node)  # remove self-loop itself

        x0, y0 = self.node_positions[node]

        # No neighbors -> default upward loop
        if not neighbors:
            return np.pi / 2

        occupied_angles = []

        for nbr in neighbors:
            x1, y1 = self.node_positions[nbr]

            dx = x1 - x0
            dy = y1 - y0

            angle = math.atan2(dy, dx)
            occupied_angles.append(angle)

        occupied_angles.sort()

        # duplicate first angle + 2π for circular gap computation
        extended = occupied_angles + [occupied_angles[0] + 2 * np.pi]

        largest_gap = -1
        best_angle = 0

        for a1, a2 in zip(occupied_angles, extended[1:]):

            gap = a2 - a1

            if gap > largest_gap:
                largest_gap = gap
                best_angle = a1 + gap / 2

        return best_angle

    """
    View
    """
    def _draw_nodes(self):
        """
        Renders all nodes on the canvas.

        Draws graph nodes using Matplotlib with fixed styling (color, size,
        borders) based on the current node position dictionary.
        """
        nx.draw_networkx_nodes(
            self.graph, 
            self.node_positions, 
            node_color='antiquewhite', 
            edgecolors='black', 
            node_size=600, 
            ax=self.canvas.axes
            )
        
    def _draw_labels(self):
        """
        Renders node labels on the graph.

        Displays text labels for each node using the current label mapping.
        """
        nx.draw_networkx_labels(
            self.graph, 
            self.node_positions, 
            labels=self.node_labels, 
            font_size=10, 
            ax=self.canvas.axes
            )
        
    def _draw_edges(self):
        """
        Draws all edges in the graph.

        Handles normal edges, bidirectional edges, and skips self-loops
        (which are drawn separately).
        """
        drawn_pairs = set()
        for u, v in self.graph.edges():
            # Self loop
            if u == v:
                continue
            # Bi-directional edge 
            if self.graph.has_edge(v, u):
                # Avoid drawing twice
                if (v, u) in drawn_pairs:
                    continue
                self._draw_bidirectional_edges(u, v)
                drawn_pairs.add((u, v))
                drawn_pairs.add((v, u))
            else:
                self._draw_normal_edge(u, v)

    def _draw_normal_edge(self, u, v):
        """
        Draws a single directed edge between two nodes.

            :param u: Source node.
            :param v: Target node.
            :type u: any
            :type v: any
        """
        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            edgelist=[(u, v)],
            edge_color=[self.edge_colors.get((u, v), self.SIGNCOLOR[0])],
            arrows=True,
            width=4,
            arrowsize=15,
            #connectionstyle='arc3,rad=0.0',
            ax=self.canvas.axes,
            )
            
    def _draw_bidirectional_edges(self, u, v):
        """
        Draws two visually separated edges for a bidirectional connection.

        Applies a perpendicular offset so that u→v and v→u edges are both
        visible without overlap.

            :param u: First node.
            :param v: Second node.
        """
        x1, y1 = self.node_positions[u]
        x2, y2 = self.node_positions[v]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return 
        
        # Perpendicular unit vector
        px = -dy / length
        py = dx / length
        offset = 0.02 
        
        # Edge u -> v
        pos1 = dict(self.node_positions)
        pos1[u] = (x1 + px * offset, y1 + py * offset)
        pos1[v] = (x2 + px * offset, y2 + py * offset)
        
        nx.draw_networkx_edges(
            self.graph,
            pos1,
            edgelist=[(u, v)],
            edge_color=[self.edge_colors.get((u, v), self.SIGNCOLOR[0])],
            arrows=True,
            width=4,
            arrowsize=15,
            #connectionstyle='arc3,rad=0.0',
            ax=self.canvas.axes,
        )

        # Edge v -> u
        pos2 = dict(self.node_positions)
        pos2[v] = (x2 - px * offset, y2 - py * offset)
        pos2[u] = (x1 - px * offset, y1 - py * offset)
        nx.draw_networkx_edges(
            self.graph,
            pos2,
            edgelist=[(v, u)],
            edge_color=[self.edge_colors.get((v, u), self.SIGNCOLOR[0])],
            arrows=True,
            width=4,
            arrowsize=15,
            #connectionstyle='arc3,rad=0.0',
            ax=self.canvas.axes,
        )
    
    import numpy as np
    import math

    def _draw_self_loops(self):
        """
        Draws self-loop edges for nodes connected to themselves.

        Uses a curved path and arrowhead to visually represent loops, and
        adjusts orientation to avoid overlap with nearby nodes.
        """
        from matplotlib.patches import PathPatch, FancyArrowPatch
        from matplotlib.path import Path

        for u, v in self.graph.edges():
            if u != v:
                continue

            x, y = self.node_positions[u]

            # 1. choose loop orientation
            angle = self._compute_self_loop_angle(u)

            radius = 0.07
            cx = x + radius * math.cos(angle)
            cy = y + radius * math.sin(angle)

            # 2. circular arc
            R = 0.05

            theta1 = 0.14 * np.pi
            theta2 = 1.9 * np.pi

            thetas = np.linspace(theta1, theta2, 60)

            xs = cx + R * np.cos(thetas)
            ys = cy + R * np.sin(thetas)

            rot = angle + np.pi

            xr = (xs - cx) * np.cos(rot) - (ys - cy) * np.sin(rot) + cx
            yr = (xs - cx) * np.sin(rot) + (ys - cy) * np.cos(rot) + cy

            # 3. path
            verts = np.column_stack([xr, yr])

            # shorten curve slightly so arrowhead sits exactly at end
            arrow_backoff = 4

            curve_verts = verts[:-arrow_backoff]

            codes = [Path.MOVETO] + [Path.LINETO] * (len(curve_verts) - 1)

            path = Path(curve_verts, codes)

            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=self.edge_colors.get((u, u), "black"),
                linewidth=3,
                capstyle='round',
                joinstyle='round',
                zorder=10
            )

            self.axes.add_patch(patch)

            # 4. arrow attached to end of loop

            # use last segment direction
            p0 = verts[-arrow_backoff - 1]
            p1 = verts[-1]

            arrow = FancyArrowPatch(
                p0,
                p1,
                arrowstyle='-|>',
                mutation_scale=20,
                color=self.edge_colors.get((u, u), "black"),
                linewidth=3,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3",
                zorder=11
            )

            self.axes.add_patch(arrow)

    def _draw_family_circles(self):
        """
        Draws small colored markers on edges to represent "family colors".
        These markers are overlaid on edges and self-loops to indicate
        additional grouping or categorization.
        BASIC_FAMILY_COLOR (white) = not drawn.
        """
        if not self.show_family_colors:
            return
        
        if not self.edge_family_colors:
            return

        for edge, color in self.edge_family_colors.items():
            if edge not in self.graph.edges():
                continue
            if tuple(color) == BASIC_FAMILY_COLOR:
                continue

            u, v = edge
            # SELF LOOP
            if u == v:
            
                x, y = self.node_positions[u]

                # same geometry as _draw_self_loops
                angle = self._compute_self_loop_angle(u)

                radius = 0.09
                cx = x + radius * math.cos(angle)
                cy = y + radius * math.sin(angle)

                R = 0.07

                theta1 = 0.15 * np.pi
                theta2 = 1.85 * np.pi

                thetas = np.linspace(theta1, theta2, 200)

                xs = cx + R * np.cos(thetas)
                ys = cy + R * np.sin(thetas)

                rot = angle + np.pi

                xr = ((xs - cx) * np.cos(rot) - (ys - cy) * np.sin(rot) + cx)

                yr = ((xs - cx) * np.sin(rot) + (ys - cy) * np.cos(rot) + cy )

                # place family marker halfway along loop
                mid_idx = len(xr) // 2

                mx = xr[mid_idx]
                my = yr[mid_idx]

                # tangent around midpoint
                tx = xr[mid_idx + 1] - xr[mid_idx - 1]
                ty = yr[mid_idx + 1] - yr[mid_idx - 1]

                # normalize
                norm = math.hypot(tx, ty)

                if norm != 0:
                    tx /= norm
                    ty /= norm

                # perpendicular (normal) vector
                nx = -ty
                ny = tx

                # move circle slightly inward toward loop center
                gap = 0.04

                mx += nx * gap
                my += ny * gap

                self.axes.add_patch(
                    plt.Circle(
                        (mx, my),
                        0.012,
                        color=color,
                        zorder=20
                    )
                )

                continue

            # NORMAL / BIDIRECTIONAL EDGE
            x1, y1 = self.node_positions[u]
            x2, y2 = self.node_positions[v]

            dx = x2 - x1
            dy = y2 - y1

            norm = math.hypot(dx, dy)
            if norm == 0:
                continue

            # midpoint
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2

            # default = centered circle
            ox = 0
            oy = 0

            # DOUBLE ASSOCIATION ONLY
            if self.graph.has_edge(v, u):

                px = -dy / norm
                py = dx / norm
                offset = 0.02

                # determine which directed edge we are drawing
                if (u, v) in self.edge_colors:   # or edge_family_colors
                    # THIS is u -> v (top edge in your drawing logic)
                    shift = -1
                else:
                    # THIS is v -> u (bottom edge)
                    shift = 1

                sx = px * offset * shift
                sy = py * offset * shift

                # midpoint of the SHIFTED edge (critical fix)
                mx = ((x1 + sx) + (x2 + sx)) / 2
                my = ((y1 + sy) + (y2 + sy)) / 2

                ox = 0
                oy = 0

            # draw family circle
            self.axes.add_patch(
                plt.Circle(
                    (mx + ox, my + oy),
                    0.012,
                    color=color,
                    zorder=20
                )
            )
    
    def _apply_zoom(self):
        """
        Applies zoom transformation to the canvas view.

        Adjusts axis limits based on the current zoom factor to scale the
        graph view around its center.
        """
        ax = self.canvas.axes
        center_x = 0.5
        center_y = 0.5
        base_width = 1.0
        base_height = 1.0
        width = base_width / self.zoom_factor
        height = base_height / self.zoom_factor
        ax.set_xlim(center_x - width / 2, center_x + width / 2)
        ax.set_ylim(center_y - height / 2, center_y + height / 2)

    def _draw_edge_preview(self, x1, y1, x2, y2):
        """
        Draws a temporary preview line for edge creation.

        Used during interactive edge creation to show a dashed line between
        source node and current mouse position.

            :param x1: Start X coordinate.
            :param y1: Start Y coordinate.
            :param x2: End X coordinate.
            :param y2: End Y coordinate.
        """
        # always clear first
        if self.edge_preview is not None:
            try:
                self.edge_preview.remove()
            except Exception:
                pass
            self.edge_preview = None

        # draw new preview
        (self.edge_preview,) = self.axes.plot(
            [x1, x2],
            [y1, y2],
            linestyle="dashed",
            linewidth=2,
            color="gray",
            alpha=0.8,
            zorder=1000,
        )

        # force immediate render (not draw_idle)
        self.canvas.draw()
    
    def _clear_edge_preview(self):
        """
        Removes the current edge preview from the canvas.

        Clears temporary visualization used during interactive edge creation.
        """
        if self.edge_preview is not None:
            try:
                self.edge_preview.remove()
            except Exception:
                pass
            self.edge_preview = None
            self.canvas.draw_idle()

    def toggle_family_colors(self):
        """
        Toggles visibility of edge family color overlays.

        Enables or disables rendering of additional edge grouping markers
        and refreshes the graph display.
        """
        self.show_family_colors = not self.show_family_colors

        print(f"Family colors visible: {self.show_family_colors}")

        self.redraw_graph()

    def redraw_graph(self):
        """
        Redraws the entire graph canvas.
    
        Clears the axes and reconstructs the full visualization including:
        nodes, labels, edges, self-loops, family markers, and zoom state.
        Preserves temporary edge preview if active.
        """
        # store preview endpoints before clearing
        preview = self.edge_preview
        preview_data = None
        if preview is not None:
            try:
                xdata = preview.get_xdata()
                ydata = preview.get_ydata()
                preview_data = (xdata, ydata)
            except Exception:
                preview_data = None

        self.axes.clear()
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.set_frame_on(False)
        self.axes.axis('off')

        self.selection_rect_patch=None

        self._draw_nodes()
        self._draw_labels()
        self._draw_edges()
        self._draw_self_loops()
        self._draw_family_circles()
        self._apply_zoom()

        # recreate preview edge properly
        if preview_data is not None:
            self.edge_preview = self.axes.plot(
                preview_data[0],
                preview_data[1],
                linestyle="dashed",
                linewidth=2,
                color="gray",
                alpha=0.7,
                zorder=1000
            )[0]
        else:
            self.edge_preview = None

        self.canvas.draw()
        
    
    """
    Controller (events)
    """
    def on_canvas_press(self, event):
        """
        Handles mouse button press events on the canvas.
            - Left-click on empty space: when dragged it starts selection rectangle or creates node
            - Left-click on node: Select/deselect node (with Shift for multi-select) for dragging
            - Right-click on node: Start or complete edge creation
            - Right-click on edge: Select edge
        """
        # cancel edge preview if clicking empty space with right click
        if event.button == 3 and self.edge_source is not None:
            nearest_node = self._get_nearest_node(event.xdata, event.ydata)
            if nearest_node is None:
                self.edge_source = None
                self._clear_edge_preview()
                return
        if event.xdata is None or event.ydata is None:
            return
        nearest_node = self._get_nearest_node(event.xdata, event.ydata)
        nearest_edge = self._get_nearest_edge(event.xdata, event.ydata)
        #right click
        if event.button == 3:
            # Right-click on node: edge creation
            if nearest_node is not None:
                if self.edge_source is None:
                    self.edge_source = nearest_node
                    print(f"Edge source selected: {nearest_node}")
                else:
                    source = self.edge_source
                    target = nearest_node
                    self._create_edge(source, target)
                    print(f"Created edge: {source} -> {target}")
                    # RESET EDGE MODE
                    self.edge_source = None
                return
            # Right-click on edge: select edge
            if nearest_edge is not None:
                self.selected_edge = self._canonical_edge(*nearest_edge)
                self.selected_nodes.clear()
                print(f"Selected edge: {nearest_edge}")
                return
        #left click on edge: select edge
        if event.button == 1 and nearest_edge is not None and nearest_node is None:
            self.selected_edge = nearest_edge
            self.selected_nodes.clear()
            print(f"Selected edge: {nearest_edge}")
            return
        #left click on node: select for dragging
        if event.button == 1 and nearest_node is not None:
            shift_pressed = event.key is not None and 'shift' in event.key.lower()
            self._select_node(nearest_node, shift_pressed)
            self.selected_edge = None
            # Set up for dragging
            self.drag_start_pos = (event.xdata, event.ydata)
            self.selection_rect_start = None
            self.dragging_nodes = True
            self.is_dragging = False
            print(f"Selected node for dragging: {nearest_node}")
            return
        #left click on empty space -> start selection rectangle or prepare to create node
        if event.button == 1:
            if nearest_node is None and nearest_edge is None:
                # Set up for selection rectangle or node creation
                self.selection_rect_start = (event.xdata, event.ydata)
                self.selected_nodes.clear()
                self.selected_edge = None
                self.dragging_nodes = False
                self.edge_source = None
                return
            
    def on_canvas_release(self, event):
        """Handles mouse button release events."""
        if event.button == 1: #if left mouse click released
            if self.is_dragging and self.tracked_positions: #If dragged, take the new positions
                self._commit_tracked_positions()
                self.tracked_positions.clear()

            xdata = event.xdata
            ydata = event.ydata
            if self.selection_rect_start is not None and xdata is not None and ydata is not None:
                # Calculate distance to check if it was a click or drag
                dx = xdata - self.selection_rect_start[0]
                dy = ydata - self.selection_rect_start[1]
                distance = math.sqrt(dx**2 + dy**2)

                # If click was on empty space with minimal movement, create a new node
                if distance < 0.03 and not self.selected_nodes:
                    self._add_new_node(self.selection_rect_start[0], self.selection_rect_start[1])
                else:
                    # Otherwise, it's a selection rectangle
                    self._select_nodes_in_rectangle(
                        self.selection_rect_start[0], self.selection_rect_start[1],
                        xdata, ydata
                    )

            # Remove selection rectangle visualization
            if self.selection_rect_patch is not None:
                self.selection_rect_patch.remove()
                self.selection_rect_patch = None
                self.canvas.draw_idle()

            self.selection_rect_start = None
            self.drag_start_pos = None
            self.dragging_nodes = False
            self.is_dragging = False

    def on_canvas_motion(self, event):
        """Handles mouse motion events: dragging nodes + edge preview."""

        if event.xdata is None or event.ydata is None:
            return

        # 1. EDGE PREVIEW (highest priority)
        if self.edge_source is not None:
            x1, y1 = self.node_positions[self.edge_source]
            x2, y2 = event.xdata, event.ydata

            self._draw_edge_preview(x1, y1, x2, y2)
            self.canvas.draw_idle()
            return  # important: prevents drag conflict
        
        # 2. SELECTION RECTANGLE PREVIEW
        if self.selection_rect_start is not None and not self.dragging_nodes:
        
            x0, y0 = self.selection_rect_start
            x1, y1 = event.xdata, event.ydata
        
            # remove old rectangle
            if self.selection_rect_patch is not None:
                self.selection_rect_patch.remove()
        
            # create dashed rectangle
            self.selection_rect_patch = Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0),
                abs(y1 - y0),
                linewidth=1.5,
                edgecolor='black',
                facecolor='lightblue',
                alpha=0.2,
                linestyle='dashed',
                zorder=999,
            )
        
            self.axes.add_patch(self.selection_rect_patch)
            self.canvas.draw_idle()

        # 3. NODE DRAGGING
        if self.drag_start_pos and self.dragging_nodes and self.selected_nodes:

            dx = event.xdata - self.drag_start_pos[0]
            dy = event.ydata - self.drag_start_pos[1]

            if abs(dx) > 0.01 or abs(dy) > 0.01:
                self.is_dragging = True
                self._track_node_positions(dx, dy)
                self.drag_start_pos = (event.xdata, event.ydata)

    def on_key_press(self, event):
        """Handle keyboard events: Delete to remove selected nodes/edges."""
        if event.key == 'delete':
            self.delete_selection()    
        if event.key == 'escape':
            self.edge_source = None
            self._clear_edge_preview()

    def _handle_double_click(self, node):
        """
        Handle double-click on a node for edge creation.
        Allows creating edges between any two nodes (including self-loops).
            :param node: The node that was double-clicked
        """
        if node is None:
            return #to guard first against weird states
        if self.double_click_node is None:
            self.double_click_node = node #First node selected
            print(f"Edge creation: First node selected: {node}")
        else:
            source = self.double_click_node   #Second node selected - create edge
            target = node
            self._create_edge(source, target)
            self.double_click_node = None
            print(f"Edge created: {source} -> {target}")
    """
    UI helpers /color tools
    """
    def _select_node(self, node, shift_pressed=False):
        """
        Select or deselect a node.
            :param node: The node to select/deselect
            :param shift_pressed: If True, add/remove from selection. If False, replace selection.
        """
        if shift_pressed:
            #Toggle selection
            if node in self.selected_nodes:
                self.selected_nodes.remove(node)
            else:
                self.selected_nodes.add(node)
        else:
            #Replace selection
            self.selected_nodes = {node}
        print(f"Selected nodes: {self.selected_nodes}")

    def _select_nodes_in_rectangle(self, x1, y1, x2, y2):
        """
        Select all nodes within a rectangle defined by two points.
            :param x1, y1: First corner
            :param x2, y2: Second corner
        """
        if self.graph is None:
            return
        #Normalize rectangle coordinates
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        positions = self.node_positions
        for node, pos in positions.items():
            if isinstance(node, int) or isinstance(node, Symbol):  #Only user nodes, not pattern nodes
                if min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y:
                    self.selected_nodes.add(node)
        print(f"Selected nodes (rectangle): {self.selected_nodes}")

    def delete_selection(self):
        """
        Delete selected nodes or edges from the graph.
            If nodes are selected, remove them and all connected edges.
            If no nodes are selected but an edge is selected, remove that edge.
            If nothing is selected, show a warning message.
        """
        if self.selected_nodes: #if select node
            self._delete_selected_nodes() #delete nodes and associated edges
            self.selected_edge=None #clear edge selection when nodes deleted
            return
        if self.selected_edge: #if edge selected
            self._delete_selected_edge() #delete edge
            self.selected_edge = None #clear edge selection after it is deleted
            return
        QMessageBox.warning(None, "No selection", "Please select a node or edge to delete") #if button pressed without selection, show error
    
    def _delete_selected_nodes(self):
        """Delete all selected nodes and their associated edges."""
        if not self.selected_nodes:
            return
        for node in list(self.selected_nodes):
            if node in self.graph:
                self.graph.remove_node(node)
            self.node_positions.pop(node, None)
            self.node_labels.pop(node, None)

            # remove edge colors involving this node
            self.edge_colors = {
                e: c for e, c in self.edge_colors.items()
                if node not in e
            }

            if hasattr(self, "edge_family_colors"):
                self.edge_family_colors = {
                    e: c for e, c in self.edge_family_colors.items()
                    if node not in e
                }
        self.selected_nodes.clear()

        self.redraw_graph()  
        self.graph_changed.emit()
    
    def _delete_selected_edge(self):
        """
        Delete the currently selected edge from the graph.
            Removes the edge from the NetworkX graph and clears any associated
            visual metadata such as edge colors and family colors. Then refreshes
            the graph and clears selection state.
        """
        if not self.selected_edge:
            return
        if self.graph.has_edge(*self.selected_edge):
            self.graph.remove_edge(*self.selected_edge)
        self.edge_colors.pop(self.selected_edge, None)
        # Also remove family color if it exists
        if hasattr(self, 'edge_family_colors'):
            self.edge_family_colors.pop(self.selected_edge, None)
        self.selected_edge= None
        self.redraw_graph()
        self.graph_changed.emit()
    
    def _get_nearest_node(self, x, y, threshold=0.05):
        """
        Find the nearest node to the given coordinates.
            :param x: X coordinate
            :param y: Y coordinate
            :param threshold: Maximum distance to consider as a click on a node
            :return: The nearest node, or None if no node is within threshold
        """
        min_dist = float('inf')
        nearest = None
        for node, pos in self.node_positions.items():
            dx= pos[0] - x
            dy= pos[1] - y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < threshold and dist < min_dist:
                nearest = node
                min_dist = dist
        return nearest
    
    def _get_nearest_edge(self, x, y, threshold=0.045):
        """
        Find the nearest edge to the given coordinates.
            :param x: X coordinate
            :param y: Y coordinate
            :param threshold: Maximum distance to consider as a click on an edge
            :return: The nearest edge as (src, tgt) tuple, or None if no edge is within threshold
        """
        candidates = []

        for src, tgt in self.graph.edges():
            # SELF LOOP HANDLING
            if src == tgt:
                x0, y0 = self.node_positions[src]
                radius = 0.08

                for t in np.linspace(0, 2 * np.pi, 30):
                    cx = x0 + radius * math.cos(t)
                    cy = y0 + radius * math.sin(t)

                    dist = math.hypot(x - cx, y - cy)

                    if dist < threshold:
                        candidates.append((dist, (src, tgt)))
                continue

            # NORMAL EDGE HANDLING
            x1, y1 = self.node_positions[src]
            x2, y2 = self.node_positions[tgt]

            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)

            if length == 0:
                continue

            # perpendicular unit vector
            px = -dy / length
            py = dx / length

            offset = 0.02

            # IMPORTANT FIX:
            # direction is ONLY based on edge direction (not tuple ordering)
            #direction = 1 if (src, tgt) in self.graph.edges() else 1
            direction = 1
            if self.graph.has_edge(tgt, src):
                direction = -1

            ox = px * offset * direction
            oy = py * offset * direction

            # apply offset ONLY to THIS directed edge
            x1o, y1o = x1 + ox, y1 + oy
            x2o, y2o = x2 + ox, y2 + oy

            dist = self._point_to_segment_distance(x, y, x1o, y1o, x2o, y2o)

            if dist < threshold:
                candidates.append((dist, (src, tgt)))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def _canonical_edge(self, u, v):
        """Return stable ordering for bidirectional pair."""
        if (u, v) in self.graph.edges() and (v, u) in self.graph.edges():
            return tuple(sorted((u, v)))
        return (u, v)
    
    def _get_event_xy(self, event):
        """Return the data coordinates for a mouse event."""
        if event.xdata is not None and event.ydata is not None:
            return event.xdata, event.ydata
        if event.inaxes is not None:
            return event.inaxes.transData.inverted().transform((event.x, event.y))
        return None, None
    
    def _move_selected_nodes(self, dx, dy):
        """
        Move all selected nodes by the given delta.
            :param dx: Change in x
            :param dy: Change in y
        """
        if not self.selected_nodes or self.graph is None:
            return
        positions = dict(self.node_positions)
        for node in self.selected_nodes:
            if node in positions:
                x, y = positions[node]
                positions[node] = (x + dx, y + dy)
        #Update the boon positions
        for node in self.selected_nodes:
            if isinstance(node, Symbol) or isinstance(node, int):
                if hasattr(self.boon, 'pos') and node in self.boon.pos:
                    x, y = self.boon.pos[node]
                    self.boon.pos[node] = (x + dx, y + dy)
        self._redraw_graph(positions)
    
    def rename_node(self):
        """
        Opens a dialog to rename the currently selected node.
        """
        if not self.selected_nodes:                                 #if no node is selected for renaming, show error
            QMessageBox.warning(None, "No Selection", "Please select a node to rename.")
            return
        if len(self.selected_nodes) > 1:                            #if multiple nodes is selected for renaming, show error
            QMessageBox.warning(None, "Multiple Selection", "Please select only one node to rename.")
            return
        node = list(self.selected_nodes)[0]                         #get the selected node
        old_name = self.node_labels.get(node, str(node))            #get current label from node_labels
        #Show input dialog
        new_name, ok = QInputDialog.getText(
            None,
            "Rename Node",
            f"Enter new name for node '{old_name}':",
            text=old_name
        )
        if ok and new_name and new_name != old_name:                #check if new name is valid and different from old name
            self.node_labels[node] = new_name                       #update the node label
            self.redraw_graph()                                     #redraw graph with new node name
            self.graph_changed.emit()                               #update graph with changes
    
    def _point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        """
        Calculate the perpendicular distance from a point to a line segment.
            :param px, py: Point coordinates
            :param x1, y1: Start of line segment
            :param x2, y2: End of line segment
            :return: Distance from point to line segment
        """
        #Vector from start to end of segment
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            #Segment is a point
            return math.hypot(px - x1, py - y1)
        #Parameter t for projection of point onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        #Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        return math.hypot(px - closest_x, py - closest_y)
    
    def zoom_in(self):
        """
        Increase the zoom level of the graph view.
            Multiplies the current zoom factor by 1.2 and redraws the graph.
        """
        self.zoom_factor *= 1.2
        self.redraw_graph()
    
    def zoom_out(self):
        """
        Decrease the zoom level of the graph view.
            Divides the current zoom factor by 1.2 and redraws the graph.
        """
        self.zoom_factor /= 1.2
        self.redraw_graph()
    
    def reset_zoom(self):
        """
        Reset the zoom level to the default value.
            Sets zoom factor back to 1.0 and redraws the graph.
        """
        self.zoom_factor = 1.0
        self.redraw_graph()
    
    def pick_edge_color(self):
        """
        Open a color picker to set the color of the selected edge.
            If no edge is selected, a warning is shown. Otherwise, the user
            selects a color which is applied to the edge.
        """
        if not self.selected_edge:
            QMessageBox.warning(self, "No edge selected", "Select an edge first.")
            return
        color = QColorDialog.getColor()
        if not color.isValid():
            return
        rgb = (color.redF(), color.greenF(), color.blueF())
        self._apply_edge_color(rgb)

    def set_edge_color_from_palette(self, color_name):
        """
        Set the color of the selected edge using a predefined palette color.
            Converts the selected hex color to RGB, stores it in edge_family_colors,
            updates the graph, and emits a change signal.
        """
        if not self.selected_edge:
            QMessageBox.warning(None, "No edge selected", "Select an edge first.")
            return
        color = QColor(color_name)
        rgb = (color.redF(), color.greenF(), color.blueF())
        if not hasattr(self, "edge_family_colors"):
            self.edge_family_colors = {}

        edge = self.selected_edge
        u, v = edge

        self.edge_family_colors[(u, v)] = rgb

        self.redraw_graph()
        self.graph_changed.emit()

    def setup_color_menu(self, parent):
        """
        Initialize the edge color palette menu.
            Defines available colors, sets initial visibility count, creates the
            QMenu container, and builds the initial palette UI.
        """
        self.color_palette = [
            ("Pastel pink", "#FFC0E7"),
            ("Pastel green", "#CAFDCB"),
            ("Pastel blue", "#CDE9FF"),
            ("Pastel yellow", "#FAF9A3"),
            ("Purple", "#ECCFFF"),
            ("Magenta", "#FF528C"),
            ("Apple Green", "#72D31C"),
            ("Teal", "#02B5C1"),
            ("Golden", "#F0CE37"),
            ("Purple", "#B66DFF"),
            ("Red", "#C6002E"),
            ("Forest green", "#095A22"),
            ("Indigo", "#0D60DD"),
            ("Orange", "#FF8400"),
            ("Violet", "#6B0E97"),
            ("Neon Pink", "#FF00F2"),
            ("Neon Green", "#4DFF00"),
            ("Cyan", "#00FFEA"),
            ("Neon Yellow", "#E5FF00"),
            ("White/None", "#FFFFFF")
            ]
        self.visible_colors = 5
        self.color_menu = QMenu(parent)
        self.build_color_palette()

    def open_color_palette(self):
        """Directly open the palette at mouse position"""
        if not hasattr(self, "color_menu"):
            return

        self.build_color_palette()
        self.color_menu.popup(QCursor().pos())

    def build_color_palette(self):
        """
        Build the color palette UI inside the menu.
            Creates a grid of color buttons, each applying a color when clicked.
            Adds controls to show more or fewer colors dynamically.
        """
        self.color_menu.clear()
        palette_widget=QWidget()
        layout=QGridLayout()
        layout.setSpacing(5)
        palette_widget.setLayout(layout)
        for i, (name, hex_color) in enumerate(self.color_palette[:self.visible_colors]):
            button=QPushButton()
            button.setFixedSize(32,32)
            button.setStyleSheet(f""" QPushButton {{ background-color: {hex_color}; border: 1px solid black; border-radius: 16px; }} """)
            button.clicked.connect(lambda _, c=hex_color: self.set_edge_color_from_palette(c))
            row=i//5
            col=i%5
            layout.addWidget(button, row, col)
        if self.visible_colors < len(self.color_palette):
            plus_button=QPushButton("+")
            plus_button.setFixedSize(32,32)
            plus_button.setStyleSheet("""QPushButton {font-size: 18px; font-weight: bold; } """)
            plus_button.clicked.connect(self.show_more_colors)
            row = self.visible_colors // 5
            col = self.visible_colors % 5
            layout.addWidget(plus_button, row, col)
        if self.visible_colors > 5:
            minus_button = QPushButton("-")
            minus_button.setFixedSize(32, 32)
            minus_button.setStyleSheet("""QPushButton {font-size: 18px; font-weight: bold;} """)
            minus_button.clicked.connect(self.show_less_colors)
            row = self.visible_colors // 5
            col = self.visible_colors 
            layout.addWidget(minus_button, row, col)
        widget_action= QWidgetAction(self.color_menu)
        widget_action.setDefaultWidget(palette_widget)
        self.color_menu.addAction(widget_action)

    def show_more_colors(self):
        """
        Increase the number of visible colors in the palette.
            Expands visible color count and rebuilds the palette UI.
        """
        self.visible_colors+=5
        self.build_color_palette()
        self.color_menu.popup(self.color_menu.pos())

    def show_less_colors(self):
        """
        Decrease the number of visible colors in the palette.
            Reduces visible color count (minimum 5) and rebuilds the palette UI.
        """
        self.visible_colors = max(5, self.visible_colors - 5)
        self.build_color_palette()
        self.color_menu.popup(self.color_menu.pos())

    def set_family_color(self, color_value):
        """
        Set a family color for the selected edge.
            Stores the color in edge_family_colors, updates the display,
            and emits a graph change signal. Requires an edge to be selected.
        """
        if not self.selected_edge:
            QMessageBox.warning(self, "No edge selected", "Select an edge first.")
            return
        color = QColor(color_value)
        rgb = (color.redF(), color.greenF(), color.blueF())
        if not hasattr(self, "edge_family_colors"):
            self.edge_family_colors = {}
        edge = self.selected_edge
        self.edge_family_colors[edge] = rgb
        # Redraw to show the family circle
        self.redraw_graph()
        self.graph_changed.emit()  # Update the model

class Network:
    """
    Conversion layer between Graph and BooN.
    This class handles transformation between the GUI graph representation and the BooN
    logical model, without UI, Qt rendering, or history management responsibilities.

        :ivar boon: Internal BooN logical model instance used for conversions.
        :type boon: BooN
    """
    def __init__(self):
        self.boon = BooN()

    def graph_to_boon(self, graph_editor, current_boon=None):
        """
        Converts a GUI graph editor structure into a BooN logical model.

        This method maps graph nodes to SymPy symbols, builds an interaction graph,
        and generates a BooN object using library functions. It also post-processes
        logical dependencies to correctly represent inhibitory interactions using Not().

            :param graph_editor: GUI graph editor containing nodes, edges, and metadata.
            :param current_boon: Optional fallback BooN model used if conversion fails.
            :return: Converted BooN model.
            :rtype: BooN
        """
        from sympy import Not, And, symbols
    
        graph = graph_editor.graph
    
        # 1. Build symbol mapping
        id_to_symbol = {}
        for node_id, label in graph_editor.node_labels.items():
            if isinstance(label, str):
                label = label.strip()
            if not label:
                label = f"x{node_id}"
            id_to_symbol[node_id] = symbols(label)
    
        # 2. Build interaction graph
        ig = nx.DiGraph()
        for symbol in id_to_symbol.values():
            ig.add_node(symbol)
    
        for u, v in graph_editor.graph.edges():
            if u not in id_to_symbol or v not in id_to_symbol:
                continue
            src_symbol = id_to_symbol[u]
            tgt_symbol = id_to_symbol[v]
            sign = graph[u][v].get("sign", 1)
            module = graph_editor.edge_modules.get((u, v), {1})
            label = graph_editor.edge_labels.get((u, v), "")
            color = graph_editor.edge_colors.get((u, v), (0, 0, 0))
            ig.add_edge(src_symbol, tgt_symbol, sign=sign, module=module, label=label, color=color)
    
        # 3. Build BooN using from_ig (library handles sign if supported)
        try:
            boon = BooN.from_ig(ig)
        except ValueError:
            if current_boon is not None:
                boon = current_boon.copy()
            else:
                boon = BooN()
    
        # FIX: Post-process desc to wrap negative-sign regulators in Not()
        for tgt_symbol in list(boon.desc.keys()):
            # Collect regulators for this target
            regulators = []
            for u, v, data in ig.edges(data=True):
                if v == tgt_symbol:
                    sign = data.get("sign", 1)
                    regulator = u if sign == 1 else Not(u)
                    regulators.append(regulator)
            if regulators:
                boon.desc[tgt_symbol] = And(*regulators)
    
        # 4. Store positions
        boon.pos = {}
        for node_id, position in graph_editor.node_positions.items():
            if node_id in id_to_symbol:
                boon.pos[id_to_symbol[node_id]] = position
    
        return boon
    
    from sympy import Not
    def boon_to_graph(self, boon, graph_editor):
        """
        Converts a BooN logical model back into a GUI graph representation.

        This method rebuilds nodes, edges, positions, labels, and visual properties
        from the BooN interaction graph and updates the graph editor accordingly.

            :param boon: BooN model to convert.
            :param graph_editor: Target GUI graph editor to populate.
            :return: None
        """
        ig = boon.interaction_graph

        graph_editor.graph.clear()

        graph_editor.node_positions.clear()
        graph_editor.node_labels.clear()
        graph_editor.edge_colors.clear()

        symbol_to_id = {}

        # 1.Nodes
        for index, node in enumerate(ig.nodes(), start=1):

            symbol_to_id[node] = index

            graph_editor.graph.add_node(index)

            graph_editor.node_labels[index] = str(node)

            if hasattr(boon, "pos") and node in boon.pos:
                graph_editor.node_positions[index] = boon.pos[node]
            else:
                graph_editor.node_positions[index] = (0.0, 0.0)

        # 2.Edges
        for src, tgt, data in ig.edges(data=True):
            src_id = symbol_to_id[src]
            tgt_id = symbol_to_id[tgt]

            graph_editor.graph.add_edge(src_id, tgt_id, sign=data.get("sign", 1))

            sign = data.get("sign", 1)

            if sign == -1:
                influence = Not(src)
            else:
                influence = src

            graph_editor.edge_colors[(src_id, tgt_id)] = graph_editor.SIGNCOLOR.get(sign, "black")

            graph_editor.edge_modules[(src_id, tgt_id)]= data.get("module", {1})
            graph_editor.edge_labels[(src_id, tgt_id)]= data.get("label", "")

        graph_editor.redraw_graph()


# DEF:  WIDGET CLASSES
class Help(QMainWindow):
    """
    Defines the Help class, a window in the application providing a user interface for
    displaying help documentation.
    This class inherits from QMainWindow and is used to load and display an HTML-based
    help file using QWebEngineView. It provides a 'Close' button to dismiss the window.
    The layout and UI components are loaded from a .ui file.
        :ivar CloseButton: The button widget used to close the help window.
        :type CloseButton: QPushButton
        :ivar web: A web engine view widget used to render and display the help HTML content.
        :type web: QWebEngineView
        :ivar WebContainer: The container widget to hold the QWebEngineView displaying the help content.
        :type WebContainer: QWidget
    """
    def __init__(self, parent=None):
        super(Help, self).__init__(parent) #initializes the parent class (QMainWindow)
        loadUi('BooNGui/help.ui', self) #loads Qt designer .ui file for layout and widgets
        self.setMinimumSize(QSize(600, 600)) #min size of help window
        self.CloseButton.clicked.connect(lambda _: self.close()) #close button from .ui file
        self.web = QWebEngineView(self) #web browser like widget
        self.WebContainer.addWidget(self.web)
        with open('BooNGui/Help.html', 'r') as f: #opens local html file
            html = f.read()
            self.web.setHtml(html)



class View(QDialog):
    """
    Represents a dialog interface for managing and displaying Boolean formulas
    in a tabular format. The interface supports operations like editing, validating,
    and converting Boolean formulas, and dynamically updates the view accordingly.
    This class is designed to be a standalone dialog that integrates with a parent
    component, using its logical formulas and variables for display and interaction.
        :ivar style: Style of the formulas displayed in the view.
        :type style: str
        :ivar parent: Reference to the parent entity using this view, typically the Boonify
            class, which contains the necessary logical formulas and variables.
        :type parent: object
        :ivar formulas: List of formula input fields linked to variables.
            Each field corresponds to a row in the table, allowing live editing of formulas.
        :type formulas: list[QLineEdit]
    """
    def __init__(self, parent=None):
        super(View, self).__init__(parent) #initializes parent class (QDialog)
        loadUi('BooNGui/view.ui', self)
        self.setGeometry(300, 300, 750, 500) #set size and position of view window
        self.style = LOGICAL  #Style of the formulas, by default: logical
        self.parent = parent  #parent = Boonify class
        self.formulas = None  #will later store input, formulas of BooN

        #STEP: Set the functions related to signals
        self.CloseButton.clicked.connect(lambda _: self.close()) #button to close view
        self.DnfButton.clicked.connect(self.convertdnf) #button to convert formulas into DNF

        #STEP: Combox Box of style.
        self.Style.activated.connect(self.cb_styling) #changes display style

        #STEP: Forbid the edition of BooNContent.
        self.BooNContent.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers) #makes table read-only but formulas are editable

        #STEP: Resize columns of the table to content.
        self.BooNContent.setColumnWidth(1, 500)

        #STEP: Fix size of the formula columns.
        header = self.BooNContent.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents) #col 0 -> small, formula type
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) #col 1 -> variable name
        header.setStretchLastSection(True)
        header.setSectionResizeMode(2, QHeaderView.Interactive) #col 2 -> formula (resizable width)

        self.initialize_view() #builds the table

    def initialize_view(self):
        """
        Initializes and populates the view with formula fields and their respective descriptions and
        attributes. This method configures a table to display rows of formulas, sets the required
        text and style for each formula, and identifies and specifies the type of logical formulation.
        """
        theboon = self.parent.boon

        #STEP: Initialize the formula fields.
        nbrow = len(theboon.desc)
        self.BooNContent.setRowCount(nbrow) #sets one row per variable 
        self.formulas = [QLineEdit() for _ in range(nbrow)]
        for f in self.formulas:
            f.editingFinished.connect(self.change_formula) #updates formula when editing is done
            f.setFrame(False)

        #STEP: Fill the table row by row
        for row, var in enumerate(theboon.desc):
            item = QTableWidgetItem(str(var)) 
            item.setTextAlignment(Qt.AlignCenter) 
            self.BooNContent.setItem(row, 1, item)  #variable name 

            self.formulas[row].setText(logic.prettyform(theboon.desc[var], style=self.style))
            self.BooNContent.setCellWidget(row, 2, self.formulas[row])  # converts formula and inserts it into the table

            if is_dnf(theboon.desc[var]):  #if type of formula: Disjunctive Normal Form
                form = "DNF"
            elif is_cnf(theboon.desc[var]): #if type formula: Conjunctive Normal Form
                form = "CNF"
            elif is_nnf(theboon.desc[var]): #if type formula: Negation Normal Form
                form = "NNF"
            else:                           #else, consider general formula
                form = "ALL"
            item = QTableWidgetItem(form)
            item.setTextAlignment(Qt.AlignHCenter)
            self.BooNContent.setItem(row, 0, item)

    def change_formula(self):
        """
        Update the BooN formula based on user input and refresh-related components.
        This method processes the formula input provided through the GUI, verifies its syntax and
        the validity of the variables involved, and updates the associated BooN data structure if
        the formula passes all checks. It also refreshes related components to reflect the changes.
        In case of errors, appropriate error messages are displayed to the user.
            :return: None
        """
        row = self.BooNContent.currentRow()  #get the current modified row
        text = self.formulas[row].text()  #get the text of the line edit formula

        try:
            formula = parse_expr(text)  # converts input/string into symbolic expression(formula)
        except SyntaxError: #if input can't be converted into valid formula, show error
            QMessageBox.critical(self, "SYNTAX ERROR", "Syntax Error.\nThe formula is not changed.\nTIP: please select the Python output form. ")
            return
        if isinstance(formula, bool):  #if formula is a bool, it has no var -> set of variables is empty
            variables = set()
        else:                          #if formula is valid, get the variables used in the formula
            variables = formula.free_symbols
        diff = variables.difference(self.parent.boon.variables) 
        if diff: #check if boon variables in formula, else show error
            QMessageBox.critical(self, "VARIABLES ERROR", f"The following variables do not exist:\n{diff}\nThe formula is not changed.")
            return
        variable = list(self.parent.boon.desc.keys())[row] #get variables of modified formulas in the row
        self.parent.boon.desc[variable] = formula  #update formulas
        self.parent.refresh() #refresh view
        self.parent.setup_design() #refresh editgraph with updated formulas

    def cb_styling(self):
        """
        Updates the current styling for the component based on the selected style and refreshes the view.
            :return: None
        """
        self.style = STYLE[self.Style.currentText()] #change the style/look of formulas
        self.initialize_view()  #Refresh the view with new style applied

    def convertdnf(self):
        """
        Converts the current BooNn into Disjunctive Normal Form (DNF)
        and refreshes the view accordingly.
            :return: None
        """
        try:
            # optional safety only (does not modify logic)
            if hasattr(self.parent.boon, "desc"):
                for k, v in self.parent.boon.desc.items():
                    if isinstance(v, str):
                        self.parent.boon.desc[k] = parse_expr(v)

            self.parent.boon.dnf() #convert all formulas into DNF
            self.initialize_view()  #Refresh the view with converted formulas
        except Exception as e:
            QMessageBox.critical(
                self,
                "DNF ERROR",
                f"DNF conversion failed:\n{str(e)}"
                )


class StableStates(QDialog):
    """
    A dialog for displaying and managing stable states in a computational model.
    This class represents a graphical interface for visualizing stable states
    of a Boolean network model. It allows users to switch between different
    display styles, such as icons or textual representation, for better
    interpretation of the stable states.
        :ivar parent: Reference to the parent widget or application component.
        :type parent: QWidget
        :ivar style: The display style for representing stable states (e.g., 'Icon Boolean').
        :type style: str
        :ivar datamodel: The data model used for organizing and displaying stable states.
        :type datamodel: QStandardItemModel
    """
    def __init__(self, parent=None):
        super(StableStates, self).__init__(parent) #initialisezs parent class (QDialog)
        loadUi('BooNGui/stablestates.ui', self)
        self.setGeometry(300, 300, 500, 700)
        self.parent = parent
        self.style = 'Icon Boolean'
        self.datamodel = None #will later store the data model for stable states
        #Button and Combo Box
        self.CloseButton.clicked.connect(lambda _: self.close())
        self.Style.activated.connect(self.cb_styling)

        #shrink the header to size
        Hheader = self.StableStatesPanel.horizontalHeader()
        Hheader.setSectionResizeMode(QHeaderView.ResizeToContents)

        self.stablestates() #build the stable states table

    def cb_styling(self):
        """
        Updates the formula display style and refreshes the view.

        This method changes how formulas are rendered (logical, Python, etc.)
        based on user selection.

            :return: None
        """
        self.style = self.Style.currentText() #change the display style
        self.stablestates()  #updates stable states view with applied style

    def stablestates(self):
        """
        Generates and sets up a data model to visualize stable states of a system.
        This method processes the stable states of a parent object's model and organizes
        them into a table-like structure using a Qt `QStandardItemModel`. Each row
        represents a variable, and each column corresponds to a stable state. The
        presentation style of the data (e.g., icons, boolean values, or integers) is
        determined by the specified `style` attribute of the object.
            :return: None
        """
        theboon = self.parent.boon #get data, current BooN of parent
        variables = theboon.variables #list of variables names in the model
        stablestates = theboon.stable_states #list of stable_states as dict (key: var name, value: its bool value in stable state)

        #Define a model of data to store stable states.
        self.datamodel = QStandardItemModel() #initialize data model for stable states
        self.datamodel.setRowCount(len(variables)) #set nb of rows = nb variables
        self.datamodel.setVerticalHeaderLabels([str(var) for var in variables]) #set var names as row headers

        #Fill the table.
        for stable in stablestates: #each stable = 1 column
            column = []
            for var in variables: #each var = 1 row, with value in state (stable[var])
                icon = QIcon()
                icon.addPixmap(QtGui.QPixmap(ICON01[stable[var]]), QtGui.QIcon.Normal, QtGui.QIcon.Off) #icon for each value
                icon.pixmap(QSize(64, 64))
                match self.style:  #define display style
                    case 'Icon':                        #icon only
                        item = QStandardItem(icon, "")
                    case 'Icon Boolean':                #icon + True/False 
                        item = QStandardItem(icon, str(stable[var]))
                    case 'Icon 0-1':                    #icon + 0/1
                        item = QStandardItem(icon, str(int(stable[var])))
                    case 'Boolean':                     #True/False only
                        item = QStandardItem(str(stable[var]))
                    case '0-1':                         #0/1 only
                        item = QStandardItem(str(int(stable[var])))
                    case _:                             #if unkown style, show error and set default style
                        item = QStandardItem("None")
                item.setTextAlignment(Qt.AlignCenter)
                column.append(item)
            self.datamodel.appendColumn(column)
        self.StableStatesPanel.setModel(self.datamodel)



class Model(QMainWindow):
    """
    A Model class for managing and visualizing network dynamics using a GUI interface.
        :ivar parent: Reference to the parent window or application.
        :type parent: Any
        :ivar mode: Represents the selected mode of dynamics (asynchronous or synchronous).
        :type mode: Enum or equivalent
        :ivar layout: Defines the network layout function to be used for visualization.
        :type layout: Callable
        :ivar canvas: Matplotlib widget for rendering the network visualization.
        :type canvas: matplotlib.backends.backend_qt5agg.FigureCanvas
    """
    def __init__(self, parent=None):
        super(Model, self).__init__(parent)
        loadUi('BooNGui/model.ui', self)
        self.setGeometry(300, 300, 600, 600)
        self.CloseButton.clicked.connect(lambda _: self.close())
        #STEP: initialize attributes
        self.parent = parent
        self.mode = boon.asynchronous
        self.layout = boon.hypercube_layout

        #STEP: Connect Matplotlib widget
        self.canvas = FigureCanvas(Figure())
        self.ModelCanvas.addWidget(self.canvas)
        self.canvas.axes = self.canvas.figure.add_subplot(111)

        #STEP: connect functions to signals.
        #Radio Button to select the mode
        self.AsynchronousButton.clicked.connect(self.rb_mode)
        self.SynchronousButton.clicked.connect(self.rb_mode)
        #Network layout Combo box
        self.NetworkLayout.activated.connect(self.cb_network_layout)

        #STEP: Modeling
        self.modeling()

    def rb_mode(self):
        """
        Determine and set the mode of operation based on user selection from the interface. This function checks the state
        of radio buttons to assign either an asynchronous or synchronous mode. The established mode is then used for
        further modeling via a further call to the `modeling` method.
            :return: None
        """
        if self.AsynchronousButton.isChecked():
            self.mode = boon.asynchronous
        elif self.SynchronousButton.isChecked():
            self.mode = boon.synchronous
        else:
            pass
        self.modeling()

    def cb_network_layout(self):
        """
        Adjusts the network layout based on the selected option and applies the corresponding
        layout algorithm to the network. The method retrieves the currently selected network
        layout from a user interface component, maps it to an appropriate algorithm, and
        configures the network's visualization layout accordingly. It also invokes an
        update via the `modeling` method to apply and reflect the changes.
            :rtype: None
        """
        layout = self.NetworkLayout.currentText()
        match layout:
            case "Hypercube":
                self.layout = boon.hypercube_layout
            case "Circular":
                self.layout = nx.circular_layout
            case "Spring":
                self.layout = nx.spring_layout
            case "Kamada Kawai":
                self.layout = nx.kamada_kawai_layout
            case "Random":
                self.layout = nx.random_layout
            case _:
                logic.errmsg("Internal Error - Unknown layout - Please contact the development team", "cb_network_layout")
        self.modeling()

    def modeling(self):
        """
        Compute the model of dynamics.
            return: None
        """
        self.canvas.axes.clear()
        self.canvas.axes.axis('off')
        model = self.parent.boon.model(mode=self.mode)
        if model.number_of_nodes() == 0:  #Empty datamodel = empty BooN
            return
        layout = self.layout(model)
        self.parent.boon.draw_model(model, pos=layout, ax=self.canvas.axes)
        self.canvas.draw_idle()



class Controllability(QMainWindow):
    """
    Controllability class for managing user interactions with the controllability
    widget of the GUI application.
    This class is responsible for initializing the controllability user interface,
    handling the interactions between destiny and observers tables, computing
    control actions based on user selections, and managing the graphical
    representation of these actions.
        :ivar parent: Parent window instance for the controllability widget.
        :type parent: QWidget
        :ivar actions: Stores the calculated control actions, if any.
        :type actions: list or None
        :ivar row: Index of the currently selected solution in control actions, if applicable.
        :type row: int or None
    """
    def __init__(self, parent=None):
        super(Controllability, self).__init__(parent)
        loadUi('BooNGui/controllability.ui', self)
        self.setGeometry(900, 300, 800, 600)
        self.parent = parent
        self.actions = None  #current control actions
        self.row = None  #row number corresponding to the selected solutions
        self.initialize_controllability()

    def initialize_controllability(self):
        """
        Initialize the controllability setup for the application.
            :param self: The class instance on which this method is invoked.
            :type self: Any
            :return: None
        """
        theboon = self.parent.boon
        variables = theboon.variables
        nbrow = len(theboon.desc)

        #STEP : set the controllability as the function of the thread
        self.parent.worker.apply(self.controllability)

        #STEP: Initialize Destiny page
        self.Destiny.setRowCount(nbrow)
        self.Destiny.resizeColumnToContents(0)  # fit size to content
        self.Destiny.horizontalHeader().setStretchLastSection(True)

        #fill the destiny table
        for row, var in enumerate(variables):
            #Add variable name
            item = QTableWidgetItem(str(var))
            item.setTextAlignment(Qt.AlignCenter)
            self.Destiny.setItem(row, 0, item)

            #Add a status for each variable
            statusbox = QComboBox(self)
            #item of the status box
            statusbox.addItems(["None", "True", "False"])  # possible status

            #insert icons for each status
            icon = QIcon(ICON01[None])  #icon for None
            icon.pixmap(QSize(64, 64))
            statusbox.setItemIcon(0, icon)
            icon = QIcon(ICON01[True])  #icon for True
            icon.pixmap(QSize(64, 64))
            statusbox.setItemIcon(1, icon)
            icon = QIcon(ICON01[False])  #icon for False
            icon.pixmap(QSize(64, 64))
            statusbox.setItemIcon(2, icon)

            #insert the status box in the table and connect it
            self.Destiny.setCellWidget(row, 1, statusbox)
            statusbox.currentTextChanged.connect(self.destiny_to_observers)

        #STEP: Initialize the observer page
        self.Observers.setRowCount(nbrow)
        self.Observers.horizontalHeader().setStretchLastSection(True)

        #fill the observer table
        for row, var in enumerate(variables):
            obschkbox = QTableWidgetItem(str(var))  #add checkbox
            obschkbox.setText(str(var))
            obschkbox.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            obschkbox.setCheckState(Qt.Unchecked)
            self.Observers.setItem(row, 0, obschkbox)

        #STEP: Define signals
        self.Observers.itemClicked.connect(self.observers_to_destiny)
        self.ControlButton.clicked.connect(self.parent.worker.run)
        self.ControlActions.clicked.connect(self.select_action)
        self.ActButton.clicked.connect(self.actupon)

        #STEP: Set the destiny page as default
        self.ControlPanel.setCurrentIndex(0)

        #STEP: Set size of columns
        #fit to content for ControlPanel
        for i in range(self.ControlPanel.count()):
            self.ControlPanel.widget(i).adjustSize()

        #fit to content for ControlActions
        header = self.ControlActions.header()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

    def destiny_to_observers(self, label: str):
        """
        Updates the check state of an item in the `Observers` table based on the
        provided label and the currently selected row in the `Destiny` table.
            :param label: A string indicating the check state condition. 
                If the label is `"None"`, the item is unchecked; otherwise, it is checked.
            :type label: str
            :return: None
        """
        row = self.Destiny.currentRow()
        item = self.Observers.item(row, 0)
        if label == "None":
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)

    def observers_to_destiny(self, chkitem):
        """
        Modify the destiny w.r.t. the observer change
            return: None
        """
        row = chkitem.row()
        if chkitem.checkState() == Qt.Unchecked:
            combobox = self.Destiny.cellWidget(row, 1)
            combobox.setCurrentText("None")

    def controllability(self):
        """
        The method calculates control actions required to achieve or avoid a defined goal 
        (or state) in a system represented by the BooN model. The method determines the applicable
        control actions by analyzing a user-specified query defining the desired or undesired state, 
        along with the possible variables that can be controlled. This process involves interpreting 
        various parameters such as observer states, query types, and logical modalities.
            :return: None
        """
        self.row = None
        theboon = self.parent.boon
        variables = list(theboon.variables)

        #STEP: Get the observers
        controlledvars = set()
        for row in range(self.Observers.rowCount()):
            item = self.Observers.item(row, 0)
            if item.checkState() == Qt.Checked:
                pass
            else:
                controlledvars.add(variables[row])

        #STEP: Get the query formula
        query = {}
        for row in range(self.Destiny.rowCount()):
            combobox = self.Destiny.cellWidget(row, 1)
            match combobox.currentText():
                case "None":
                    pass
                case "True":
                    query.update({variables[row]: True})
                case "False":
                    query.update({variables[row]: False})

        #STEP: Convert the state profiles into minterm
        formula = SOPform(query.keys(), [query])
        #Check whether the query must be reached or avoid.
        match self.QueryType.currentText():
            case "Reach":
                pass
            case "Avoid":
                formula = Not(formula)

        #STEP: Add control
        boonctrl = theboon.copy()
        boonctrl.control(controlledvars, controlledvars)

        #STEP: Interpret the modality of the query (possibility, necessity).
        if self.Possibility.isChecked():  #Possibility
            possibility = boonctrl.possibly(formula)
        else:
            possibility = True
        if self.Necessity.isChecked():  #Necessity
            necessity = boonctrl.necessary(formula, trace=self.Trace)
        else:
            necessity = True
        destiny = And(possibility, necessity)

        #STEP: Destify the controlled BooN and transform the solutions into control actions (var, Boolean Value)
        core = boonctrl.destify(destiny, trace=self.Trace, solver=LPSOLVER)
        self.actions = boon.core2actions(core)

        #STEP: Define the tree model to show the resulting actions.
        treemodel = QStandardItemModel(0, 2)  #Add 2 columns
        treemodel.setHeaderData(0, Qt.Horizontal, "Variable")
        treemodel.setHeaderData(1, Qt.Horizontal, "Boolean value")

        root = treemodel.invisibleRootItem()
        match self.actions:
            case []:  #No actions
                item = QStandardItem("No action found.")
                root.appendRow(item)
            case [[]]:  #The destiny profile is already obtained
                item = QStandardItem("The marking profile already exists.")
                root.appendRow(item)
            case _:  #Compute the control actions.
                for i, actions in enumerate(self.actions, 1):
                    #create the root solution
                    rootactions = QStandardItem("Solution {:2d}".format(i))
                    #add the control actions of this solution
                    for action in actions:
                        #a control action is displayed as: variable + Boolean icon + Boolean value
                        variable = QStandardItem(str(action[0]))
                        icon = QIcon(ICON01[action[1]])
                        icon.pixmap(QSize(64, 64))
                        value = QStandardItem(icon, str(action[1]))
                        rootactions.appendRow([variable, value])
                    #append the solution to the tree model
                    root.appendRow(rootactions)
        self.ControlActions.setModel(treemodel)  #set the data model to tree widget enabling its display.
        self.ControlActions.expandAll()

    def select_action(self, arg):
        """
        Keep the selection solution
            return: None
        """
        self.row = arg.parent().row() if arg.parent().row() > -1 else arg.row()

    def actupon(self):
        """
        Apply the selection actions on the BooN.
            return: None
        """
        if self.row is not None and self.actions:
            for action in self.actions[self.row]:
                (variable, value) = action
                self.parent.boon.desc[variable] = value
            self.parent.add_history()
            self.parent.setup_design()
            self.parent.refresh()
            self.close()



class Threader(QObject):
    """
    Threader class for managing application execution within a separate thread.
    This class is designed to separate the execution of a provided application
    function into its own thread using PyQt's threading mechanism. The class
    provides functionality to start, switch, and terminate the application
    running within the thread effectively.
        :ivar finished: Signal emitted when the thread's application completes execution.
        :ivar app: The callable application to be executed within the thread.
        :type app: Callable
        :ivar thread: The QThread instance used to run the application in a separate thread.
        :type thread: QThread
    """
    finished = pyqtSignal()
    def __init__(self, app=lambda: None):
        """
        Represents a custom asynchronous functionality encapsulated in a QThread.
        This class initializes with a callable app function, assigns it to an internal
        property, and starts a new thread for its execution.
        Attributes:
            app (Callable[[], Any]): A callable function assigned to the class instance.
            thread (QThread): The thread in which the object operates.
            :param app: A callable function that serves as the application's main function.
                Defaults to a no-op lambda function.
            :type app: Callable[[], Any]
        """
        super().__init__()
        self.app = app

        #Create the thread
        self.thread = QThread()
        self.thread.started.connect(self.run)
        self.moveToThread(self.thread)
        self.thread.start()

    def run(self):
        """
        Defines the run function to execute the primary application logic and signal
        completion. The function encompasses two main operations: invoking the
        application logic and signaling the end of the process.
        """
        try:
            self.app()  #run the application
        finally:
            self.finished.emit()  #Emit the end signal

    def apply(self, app):
        """
        Handles the application of a given app instance to a particular object by
        assigning the provided app to the instance attribute.
            :param app: The application instance to be applied.
            :return: None
        """
        self.app = app

    def quit(self):
        """
        Attempts to terminate the thread execution in an orderly manner. This method
        leverages the `quit` function of the thread instance to signal and ensure
        termination of its event loop.
            :return: None
        """
        self.thread.quit()
        self.thread.wait()



#DEF:  MAIN
if __name__ == "__main__":
    app = QApplication(sys.argv)
    boonify = Boonify()
    #callback = boonify.canvas.mpl_connect('draw_event', lambda _: boonify.graph_editor.design())
    boonify.show()
    sys.exit(app.exec_())
