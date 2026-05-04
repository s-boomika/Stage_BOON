#Graphical interface for BooN design and analysis
#Author: Franck Delaplace
#Creation date: February 2024

# In comments:
#DEF means definition which is a code part gathering functions related to a process or an object definition,
#STEP means main steps
#WARNING is a warning.
#These terms can be used to color comments in PyCharm or else.

# Standard library imports
import sys
import os
import math
import re

# Third-party imports: import functions only used 
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from netgraph import EditableGraph #ne pas utiliser editgraph
from sympy import SOPform, symbols
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import is_cnf, is_dnf, is_nnf
from sympy.logic.boolalg import And, Not
from sympy.parsing.sympy_parser import parse_expr
from pulp import PULP_CBC_CMD

#PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.uic import loadUi


from turtle import color, pos
from matplotlib import colors
from matplotlib.pylab import norm
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

#Node names for the pattern network
REG: str = '\u03b1'  #lambda
POS: str = '\u03b2'  #alpha
NEG: str = '\u03bb'  #beta

#Integer regular expression
INTPAT: str = r"\s*-?[0-9]+\s*"


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
        self.filename = ""  #Current filename
        self.history = [None] * HSIZE  #History
        self.hindex = 0  #Index of the last BooN added in the history.
        self.hupdate = False  #Flag determining whether the history is updated.
        self.saved = True  #Flag determining whether the current BooN is saved.
        self.QView = None  #Widget of the View
        self.QStableStates = None  #Widget of the stable states
        self.QModel = None  #Widget of the dynamics model
        self.QControllability = None  #Widget of the controllability
        self.editgraph = None  #Graph for the edition
        self.disablecallback = True  #Flag indicating whether the BooN design callback function is enabled, initially disabled (True) because the editgraph is not set up.
        self.designsize = 2.  #Size related to the EditableGraph and used for many parameters. It is modified when the window is rescaled to let the nodes and edges width invariant.
        
        #Selection and edge creation state
        self.selected_nodes = set()  #Currently selected nodes
        self.selected_edge = None
        self.double_click_node = None  #First node in double-click edge creation
        self.double_click_timer = None  #Timer for tracking double-clicks
        self.drag_start_pos = None  #Position where drag started
        self.selection_rect_start = None  #Start position for selection rectangle
        self.is_dragging = False  #Whether we're currently dragging nodes
        self.tracked_positions = {}  #Positions being tracked during drag (not yet committed)
        self.DeleteButton.clicked.connect(self.delete_selection)
        self.RenameNodeButton.clicked.connect(self.rename_node) #Rename node when button clicked on and then clicked on a node selected

        self.display_saved_flag()  #Show the save flag in the status bar

        self.sign_menu = QMenu(self)

        pos_action = self.sign_menu.addAction("Positive")
        neg_action = self.sign_menu.addAction("Negative")
        #neu_action = self.sign_menu.addAction("Neutral")

        pos_action.triggered.connect(lambda: self.change_edge_sign(1))
        neg_action.triggered.connect(lambda: self.change_edge_sign(-1))
        #neu_action.triggered.connect(lambda: self.change_edge_sign(0))

        self.ChangeEdgeSignButton.setMenu(self.sign_menu)

        if not hasattr(self, "ColorEdgeButton"):
            # Create fallback button if the UI does not define it
            self.ColorEdgeButton = QPushButton("Set Family Color")
            self.DesignCanvas.addWidget(self.ColorEdgeButton)

        # Setup color menu
        self.setup_color_menu()

        #circles for family assignment
        self.family_circles = {}
        self.edge_family_colors = {}

        self.ZoomInButton.clicked.connect(self.zoom_in)
        self.ZoomOutButton.clicked.connect(self.zoom_out)

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

        #STEP: Initialize the Canvas for network design
        fig = plt.figure()  #plt.figure is necessary to create a manager
        manager = fig.canvas.manager
        self.canvas = FigureCanvas(fig)
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.canvas.figure.subplots_adjust(left=0, bottom=0, right=1, top=1)  # Adjust the window s.t. The network fully fills it.
        self.canvas.figure.canvas.manager = manager  #Assign the manager in the canvas to be accessible by EditGraph.

        #STEP: Enable key_press_event events for using EditGraph:
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()

        #STEP: Insert the network design figure in the main window.
        self.DesignCanvas.addWidget(self.canvas)

        #STEP: set up network design
        self.setup_design()
        
        #STEP: Connect mouse and keyboard events
        self.canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

    #DEF: NETWORK DESIGN
    def setup_design(self):
        """
        Sets up the design of the interaction graph and its visual representation. The method configures
        node and edge properties, styles, positions, and visual elements on the canvas. It integrates
        information from an existing interaction graph and generates an editable representation of the
        network for visualization.

        Attributes that affect the design are updated during this process, and the design callback is
        temporarily disabled to avoid interference.
        """
        self.disablecallback = True  #Disable the design callback during the set-up.

        #STEP: Creation of the pattern-network used for edge styling.
        #To prevent its inclusion in the BooN, the variable names are strings while the other nodes are integers or symbols.

        #network structure:  NEG <--[-]-- REG --[+]--> POS
        g = nx.DiGraph([(REG, POS), (REG, NEG)])
        edge_color = {(REG, POS): SIGNCOLOR[1], (REG, NEG): SIGNCOLOR[-1], (NEG, NEG): SIGNCOLOR[-1], (POS, POS): SIGNCOLOR[1]}
        positions = {NEG: (0.25, 0.045), REG: (0.5, 0.045), POS: (0.75, 0.045)}

        #STEP: Add  the interaction graph of the current BooN.
        ig = self.boon.interaction_graph  #Get the IG.
        g.add_nodes_from(ig.nodes())
        g.add_edges_from(ig.edges())

        #STEP: Find the edge color from signs.
        colors = nx.get_edge_attributes(ig, 'color')
        edge_color.update({edge: colors.get(edge, SIGNCOLOR[dasigns[edge]]) for edge in ig.edges()})  # Transform sign in color.
        positions.update(self.boon.pos)  #Complete the position.

        #STEP: Convert the module ids to string labeling the edge.
        modules = nx.get_edge_attributes(ig, 'module')
        modules = {edge: " ".join([str(module) for module in modules[edge]]) for edge in modules}

        #STEP: Define the editgraph from the information previously extracted.
        self.canvas.axes.clear()
        self.canvas.axes.set_xlim(-1000, 1000)
        #WARNING : the definition of EditableGraph is the same as the definition in resizeEvent function. Any modification of one must be reported to the other.
        self.editgraph = EditableGraph(g,
                                       node_labels=True,
                                       node_color='antiquewhite',
                                       node_edge_color='black',
                                       node_label_fontdict=dict(family='sans-serif', color='black', weight='semibold', fontsize=11),
                                       node_layout=positions,
                                       node_size=self.designsize,
                                       node_edge_width=self.designsize / 4,
                                       node_label_offset=(0., 0.025 * self.designsize),
                                       edge_width=self.designsize / 2,
                                       arrows=True,
                                       edge_labels=modules,
                                       edge_label_position=0.75,
                                       edge_label_fontdict=dict(family='sans-serif', fontweight='bold', fontsize=8, color='royalblue'),
                                       edge_color=edge_color,
                                       ax=self.canvas.axes)
        self.disablecallback = False  #Enable the design callback.

    def design(self):
        """
        Designs the BooN based on the events captured from the EditableGraph.
        This function establishes consistency in the nodes and edges of the graph, processes symbolic
        representation, determines signs and modularity of edges, and converts the interaction
        graph (IG) to a BooN object for further utilization.
        """
        if self.disablecallback: 
            return

        try:
            #DEF: Definition of the interaction graph of the network from the drawing.
            idvar = {idt: symbols(text.get_text()) for idt, text in self.editgraph.node_label_artists.items() if isinstance(idt, int | Symbol)}

            def symbolic(edge):
                """set edges in symbolic form where node labels are symbols"""
                return idvar[edge[0]], idvar[edge[1]]

            edges = {(src, tgt) for src, tgt in self.editgraph.edge_artists.keys() if src in idvar and tgt in idvar}

            edit_pos = self.editgraph.node_positions
            pos = {idvar[idt]: edit_pos[idt] for idt in idvar}

            edge_attributes = self.editgraph.edge_artists
            signs = {}
            for edge in edges:
                #fallback if no stored sign
                signs[edge] = self.boon.interaction_graph.get_edge_data(*edge, {}).get("sign", 0)

            edge_labels = self.editgraph.edge_label_artists
            modules = {}
            for edge in edges:
                try:
                    label = edge_labels[edge].get_text()
                    modularity = {int(module) for module in list(label.split(" ")) if re.match(INTPAT, module)}

                    if modularity:
                        match signs[edge]:
                            case 1:  #Positive
                                modularity = {abs(module) for module in modularity}
                            case -1:  #Negative
                                modularity = {-abs(module) for module in modularity}
                            case 0:  #undefined (both)
                                pass
                        modules.update({symbolic(edge): modularity})
                    else:
                        modules.update({symbolic(edge): {signs[edge]}})

                except KeyError:  #No edge labels = no modules
                    modules.update({symbolic(edge): {signs[edge]}})

            #Convert edges symbolically for signs
            signs = {symbolic(edge): signs[edge] for edge in signs}

            #Define the ig from the editable graph
            ig = nx.DiGraph()
            ig.add_nodes_from(idvar.values())
            for edge in signs:
                ig.add_edge(edge[0], edge[1],
                            sign=signs[edge],
                            module=modules[edge],
                            color=edge_attributes[edge].get_facecolor()[0:3])

            #Convert the ig to BooN
            try:
                self.boon = BooN.from_ig(ig)
            except ValueError:
                pass

            #Store positions
            self.boon.pos = pos

            #Update history and refresh open windows
            self.add_history()
            if self.hupdate:
                self.refresh()

        except Exception as e:
            #Catch any errors during design update
            print(f"Design error (continuing): {e}")
            import traceback
            traceback.print_exc()

    def on_canvas_press(self, event):
        """
        Handles mouse button press events on the canvas.
            - Left-click on empty space: when dragged it starts selection rectangle or creates node
            - Left-click on node: Select/deselect node (with Shift for multi-select)
            - Double-click on node: Start edge creation or complete it
        """

        #reset edge creation if user clicks anything other than right double-click
        if not (event.dblclick and event.button ==3):
            self.double_click_node = None
        #ignore clicks outside canvas
        if event.xdata is None or event.ydata is None:
            return
        
        #For node selection, finding the nearest node 
        nearest_node = self._get_nearest_node(event.xdata, event.ydata)
        if nearest_node is not None:
            shift_pressed = event.key == 'shift'
            self._select_node(nearest_node, shift_pressed) #for multi-selection
            self.selected_edge = None #clear edge selection if clicked on a node
            self.drag_start_pos = (event.xdata, event.ydata) #store position for drag feature
            return

        #For edge selection, find nearest edge
        edge = self._get_nearest_edge(event.xdata, event.ydata)
        if edge is not None:
            self.selected_edge = edge
            self.selected_nodes.clear()  #clear node selection if clicked on an edge
            print(f"Selected edge: {edge}")
            return

        #For double-click first
        if event.dblclick:
            if event.button !=3: #ignore all non right double click
                return
            nearest_node = self._get_nearest_node(event.xdata, event.ydata) #get nearest node when clicked
            if nearest_node is not None:
                self._handle_double_click(nearest_node) #for double click edge creation
            return #stops all further processings on double-clicks

        #For normal left mouse button
        if event.button == 1: #1 is for left-click
            nearest_node = self._get_nearest_node(event.xdata, event.ydata) #get nearest node when clicked
            if nearest_node is not None: #when a single left click detected
                shift_pressed = event.key == 'shift' #selected with shift
                self._select_node(nearest_node, shift_pressed) #or selected without shift
                self.drag_start_pos = (event.xdata, event.ydata) #can be dragged when selected
            else: #when clicked on empty space, left click starts selection rectangle/multi-selection
                if not event.key == 'shift': 
                    self.selected_nodes.clear()  #Clear selection if not holding Shift
                self.selection_rect_start = (event.xdata, event.ydata) #store position for selection rectangle
                self.drag_start_pos = (event.xdata, event.ydata) #start drag from here if mouse moves

    def on_canvas_release(self, event):
        """Handles mouse button release events."""
        if event.button == 1: #if left mouse click released
            if self.is_dragging and self.tracked_positions: #If dragged, take the new positions
                self._commit_tracked_positions()
                self.tracked_positions.clear()
            #Check if we should create a new node (click on empty space with minimal movement)
            if (self.selection_rect_start is not None and event.xdata is not None and event.ydata is not None):
                #Calculate distance to check if it was a click or drag
                dx = event.xdata - self.selection_rect_start[0]
                dy = event.ydata - self.selection_rect_start[1]
                distance = math.sqrt(dx**2 + dy**2)
                #If click was on empty space with minimal movement, create a new node
                if distance < 0.01 and not self.selected_nodes:
                    self._add_new_node(self.selection_rect_start[0], self.selection_rect_start[1])
                else:
                    #Otherwise, it's a selection rectangle
                    self._select_nodes_in_rectangle(
                        self.selection_rect_start[0], self.selection_rect_start[1],
                        event.xdata, event.ydata
                    )
            self.selection_rect_start = None
            self.drag_start_pos = None
            self.is_dragging = False

    def on_canvas_motion(self, event):
        """Handles mouse motion events: used for dragging nodes."""
        if event.xdata is None or event.ydata is None or not self.drag_start_pos: #ignore if outside canvas or no drag started
            return
        
        if event.button == 1 and self.selected_nodes and not self.selection_rect_start: #if left clicked and node selected without a selection rectangle made
            #Drag selected nodes
            dx = event.xdata - self.drag_start_pos[0]
            dy = event.ydata - self.drag_start_pos[1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:  #Threshold to distinguish from click
                self.is_dragging = True
                #Just track positions-don't redraw until release
                self._track_node_positions(dx, dy)
                self.drag_start_pos = (event.xdata, event.ydata)

    def on_key_press(self, event):
        """Handle keyboard events: Delete to remove selected nodes/edges."""
        if event.key == 'delete':
            self.delete_selection()

    def _create_edge(self, source, target):
        """
        Create an edge between two nodes (including self-loops).
            :param source: Source node
            :param target: Target node
        """
        try:
            #Get current state
            positions = dict(self.editgraph.node_positions)
            edge_color = {edge: self.editgraph.edge_artists[edge].get_facecolor()[0:3] for edge in self.editgraph.edge_artists}
            modules = {edge: self.editgraph.edge_label_artists[edge].get_text() for edge in self.editgraph.edge_label_artists}
            node_labels = {idt: text.get_text() for idt, text in self.editgraph.node_label_artists.items()}

            #Copy graph and add edge
            g = nx.DiGraph()
            g.add_nodes_from(self.editgraph.nodes)
            g.add_edges_from(self.editgraph.edges)
            
            #Add the new edge
            if not g.has_edge(source, target):
                g.add_edge(source, target)
                # Initialize edge attributes
                edge_color[(source, target)] = SIGNCOLOR[0]  #Default to neutral/undefined sign
                modules[(source, target)] = ""  #No modules initially
                print(f"Edge created: {source} -> {target}")
            else:
                print(f"Edge {source} -> {target} already exists")
                return

            #Recreate EditableGraph
            self._redraw_graph_with_edges(g, positions, edge_color, modules, node_labels)
            
            #Trigger design update (with error handling)
            try:
                self.design()
            except Exception as e:
                print(f"Design error after edge creation: {e}")
        
        except Exception as e:
            print(f"Error creating edge: {e}")
            import traceback
            traceback.print_exc()

    def _add_new_node(self, x, y):
        """
        Add a new node at the specified position.
            :param x: X coordinate
            :param y: Y coordinate
        """
        #Find next integer id for the new node
        nodes = list(self.editgraph.nodes)
        int_nodes = [n for n in nodes if isinstance(n, int)]
        new_id = max(int_nodes) + 1 if int_nodes else 1

        #Get current state
        positions = dict(self.editgraph.node_positions)
        positions[new_id] = (x, y)

        edge_color = {edge: self.editgraph.edge_artists[edge].get_facecolor()[0:3] for edge in self.editgraph.edge_artists}
        modules = {edge: self.editgraph.edge_label_artists[edge].get_text() for edge in self.editgraph.edge_label_artists}
        node_labels = {idt: text.get_text() for idt, text in self.editgraph.node_label_artists.items()}
        node_labels[new_id] = str(new_id)

        #Copy graph and add node
        g = nx.DiGraph()
        g.add_nodes_from(self.editgraph.nodes)
        g.add_edges_from(self.editgraph.edges)
        g.add_node(new_id)

        #Recreate EditableGraph
        self._redraw_graph_with_edges(g, positions, edge_color, modules, node_labels)

    def _delete_selected_nodes(self):
        """Delete all selected nodes and their associated edges."""
        if not self.selected_nodes or self.editgraph is None:
            return
        
        #Get current state
        positions = dict(self.editgraph.node_positions)
        edge_color = {edge: self.editgraph.edge_artists[edge].get_facecolor()[0:3] for edge in self.editgraph.edge_artists}
        modules = {edge: self.editgraph.edge_label_artists[edge].get_text() for edge in self.editgraph.edge_label_artists}
        node_labels = {idt: text.get_text() for idt, text in self.editgraph.node_label_artists.items()}

        #Create new graph without selected nodes
        g = nx.DiGraph()
        remaining_nodes = [n for n in self.editgraph.nodes if n not in self.selected_nodes]
        g.add_nodes_from(remaining_nodes)
        
        #Add edges that don't involve selected nodes
        for src, tgt in self.editgraph.edges:
            if src not in self.selected_nodes and tgt not in self.selected_nodes:
                g.add_edge(src, tgt)
        
        #Clean up positions and labels
        positions = {n: positions[n] for n in remaining_nodes if n in positions}
        node_labels = {n: node_labels[n] for n in remaining_nodes if n in node_labels}
        
        #Clean up edge attributes
        edge_color = {edge: color for edge, color in edge_color.items() if edge[0] in remaining_nodes and edge[1] in remaining_nodes}
        modules = {edge: mod for edge, mod in modules.items() if edge[0] in remaining_nodes and edge[1] in remaining_nodes}
        
        self.selected_nodes.clear()
        self._redraw_graph_with_edges(g, positions, edge_color, modules, node_labels)
        self.design()
        print(f"Deleted nodes and associated edges")

    def _delete_selected_edge(self):
        if not self.selected_edge or self.editgraph is None:
            return

        edge_to_remove = self.selected_edge
        #Current state
        positions = dict(self.editgraph.node_positions)
        edge_color = {edge: self.editgraph.edge_artists[edge].get_facecolor()[0:3]
                      for edge in self.editgraph.edge_artists}
        modules = {edge: self.editgraph.edge_label_artists[edge].get_text()
                   for edge in self.editgraph.edge_label_artists}
        node_labels = {idt: text.get_text()
                       for idt, text in self.editgraph.node_label_artists.items()}

        #Rebuild graph without the edge
        g = nx.DiGraph()
        g.add_nodes_from(self.editgraph.nodes)

        for edge in self.editgraph.edges:
            if edge != edge_to_remove:
                g.add_edge(*edge)
        #Remove edge attributes
        edge_color.pop(edge_to_remove, None)
        modules.pop(edge_to_remove, None)

        self._redraw_graph_with_edges(g, positions, edge_color, modules, node_labels)
        #Update BooN
        self.design()

        print(f"Deleted edge: {edge_to_remove}")

    def _get_nearest_node(self, x, y, threshold=0.05):
        """
        Find the nearest node to the given coordinates.
            :param x: X coordinate
            :param y: Y coordinate
            :param threshold: Maximum distance to consider as a click on a node
            :return: The nearest node, or None if no node is within threshold
        """
        if self.editgraph is None:
            return None
        
        positions = self.editgraph.node_positions
        min_dist = float('inf')
        nearest = None
        
        for node, pos in positions.items():
            dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                nearest = node
        return nearest

    def _get_nearest_edge(self, x, y, threshold=0.05):
        """
        Find the nearest edge to the given coordinates.
            :param x: X coordinate
            :param y: Y coordinate
            :param threshold: Maximum distance to consider as a click on an edge
            :return: The nearest edge as (src, tgt) tuple, or None if no edge is within threshold
        """
        if self.editgraph is None:
            return None
        
        min_dist = float('inf')
        nearest_edge = None
        
        for (src, tgt), artist in self.editgraph.edge_artists.items():
            path = artist.get_path().vertices
            for i in range(len(path) - 1):
                px1, py1 = path[i]
                px2, py2 = path[i + 1]
                #Calculate perpendicular distance from point to line segment
                dist = self._point_to_segment_distance(x, y, px1, py1, px2, py2)
                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    nearest_edge = (src, tgt)
        return nearest_edge
    
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
        if self.editgraph is None:
            return
        
        #Normalize rectangle coordinates
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        positions = self.editgraph.node_positions
        for node, pos in positions.items():
            if isinstance(node, int) or isinstance(node, Symbol):  #Only user nodes, not pattern nodes
                if min_x <= pos[0] <= max_x and min_y <= pos[1] <= max_y:
                    self.selected_nodes.add(node)
        print(f"Selected nodes (rectangle): {self.selected_nodes}")

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

    def _track_node_positions(self, dx, dy):
        """
        Track position changes without redrawing (used during drag).
        Stores positions for later commit.
            :param dx: Change in x
            :param dy: Change in y
        """
        if not self.selected_nodes or self.editgraph is None:
            return
        #Initialize tracked positions from current if first time
        if not self.tracked_positions:
            positions = dict(self.editgraph.node_positions)
            for node in self.selected_nodes:
                if node in positions:
                    self.tracked_positions[node] = positions[node]
        #Update tracked positions
        for node in self.selected_nodes:
            if node in self.tracked_positions:
                x, y = self.tracked_positions[node]
                self.tracked_positions[node] = (x + dx, y + dy)

    def _commit_tracked_positions(self):
        """
        Commit tracked positions to both editgraph and boon.
        Triggers a single redraw and design update.
        """
        if not self.tracked_positions or self.editgraph is None:
            return
        #Update editgraph positions
        positions = dict(self.editgraph.node_positions)
        for node, pos in self.tracked_positions.items():
            positions[node] = pos
        #Update boon positions
        for node, pos in self.tracked_positions.items():
            if isinstance(node, Symbol) or isinstance(node, int):
                if hasattr(self.boon, 'pos') and node in self.boon.pos:
                    self.boon.pos[node] = pos
        #Single redraw
        self._redraw_graph(positions)
        # redraw family circles
        if hasattr(self, "edge_family_colors"):
            for edge, color in self.edge_family_colors.items():
                self._draw_family_circle(edge, color)
        self.canvas.draw()
        #Single design update
        try:
            self.design()
        except Exception as e:
            print(f"Design update error: {e}")

    def _move_selected_nodes(self, dx, dy):
        """
        Move all selected nodes by the given delta.
            :param dx: Change in x
            :param dy: Change in y
        """
        if not self.selected_nodes or self.editgraph is None:
            return
        positions = dict(self.editgraph.node_positions)
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

    def _redraw_graph(self, positions):
        """Redraw the graph with updated node positions."""
        if self.editgraph is None: #if there is no graph to redraw, do nothing
            return
        
        #store current edges colors, edge modules(labels) and labels of nodes 
        edge_color = {edge: self.editgraph.edge_artists[edge].get_facecolor()[0:3] for edge in self.editgraph.edge_artists}
        modules = {edge: self.editgraph.edge_label_artists[edge].get_text() for edge in self.editgraph.edge_label_artists}
        node_labels = {idt: text.get_text() for idt, text in self.editgraph.node_label_artists.items()}
        
        #create a new graph with networkx (nx) to copy nodes and edges from editgraph
        g = nx.DiGraph()
        g.add_nodes_from(self.editgraph.nodes) #copies nodes to new graph (g)
        g.add_edges_from(self.editgraph.edges) #copies edges to new graph (g)
        
        #redraw graph with new positions, stored edge colors, edge modules and node labels
        self._redraw_graph_with_edges(g, positions, edge_color, modules, node_labels)

    def _redraw_graph_with_edges(self, g, positions, edge_color, modules, node_labels):
        """Helper method to redraw the entire graph."""
        self.disablecallback = True #disable design callback during redraw to avoid interferences
        self.canvas.axes.clear() #clear canvas before redraw
        #Rebuild graph with EditableGraph class, with styling specified
        self.editgraph = EditableGraph(g,
                                       node_labels=node_labels,
                                       node_color='antiquewhite',
                                       node_edge_color='black',
                                       node_label_fontdict=dict(family='sans-serif', color='black', weight='semibold', fontsize=11),
                                       node_layout=positions, #use updated node positions 
                                       node_size=self.designsize,
                                       node_edge_width=self.designsize / 4,
                                       node_label_offset=(0., 0.025 * self.designsize),
                                       edge_width=self.designsize / 2,
                                       arrows=True,
                                       edge_labels=modules,
                                       edge_label_position=0.75,
                                       edge_label_fontdict=dict(family='sans-serif', fontweight='bold', fontsize=8, color='royalblue'),
                                       edge_color=edge_color,
                                       ax=self.canvas.axes)
        self.disablecallback = False #enable callback after redraw
        self._clear_family_circles() #clear old circles first
        
        # redraw family circles after graph refresh, with current positions
        if hasattr(self, "edge_family_colors"):
            for edge, color in self.edge_family_colors.items():
                if edge in g.edges: 
                    self._draw_family_circle(edge, color) 

        self.canvas.draw() #updates the canvas, by displaying the new graph

    def rename_node(self):
        """
        Opens a dialog to rename the currently selected node.
        """
        if not self.selected_nodes: #if no node is selected for renaming, show error
            QMessageBox.warning(self, "No Selection", "Please select a node to rename.")
            return
        if len(self.selected_nodes) > 1: #if multiple nodes is selected for renaming, show error
            QMessageBox.warning(self, "Multiple Selection", "Please select only one node to rename.")
            return
    
        node = list(self.selected_nodes)[0] #get the selected node
        old_name = str(node) #get current name of node
    
        #Show input dialog
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Node",
            f"Enter new name for node '{old_name}':",
            text=old_name
        )
    
        if ok and new_name and new_name != old_name: #check if new name is valid and different from old name
            if node in self.editgraph.node_label_artists: #if node has label, update it in editgraph (display canvas)
                self.editgraph.node_label_artists[node].set_text(new_name)

            self._redraw_graph(self.editgraph.node_positions) #redraw graph with new node name
            self.design() #update graph with changes

    def delete_selection(self):
        if self.selected_nodes: #if select node
            self._delete_selected_nodes() #delete nodes and associated edges
            self.selected_edge=None #clear edge selection when nodes deleted
            return
        if self.selected_edge: #if edge selected
            self._delete_selected_edge() #delete edge
            self.selected_edge = None #clear edge selection after it is deleted
            return
        QMessageBox.warning(self, "No selection", "Please select a node or edge to delete") #if button pressed without selection, show error

    def change_edge_sign(self, sign):
        if not self.selected_edge: #if no edge selected, show error
            QMessageBox.warning(self, "No edge selected", "Please click an edge first.")
            return

        edge = self.selected_edge #store selected edge

        #Update color of edge
        color = SIGNCOLOR[sign] #get color of sign assigned to edge
        self.editgraph.edge_artists[edge].set_facecolor(color) #update visual color of edge in editgraph (canvas)

        #Optional: update label
        if edge in self.editgraph.edge_label_artists: #if edge has label, update it to sign assigned
            self.editgraph.edge_label_artists[edge].set_text(str(sign)) 

        self.canvas.draw() #refresh canvas to show changes

        self.design() #update design to have changes in BooN model

    def _apply_edge_color(self, rgb):
        edge = self.selected_edge
        if edge not in self.editgraph.edge_artists:
            return
        #Update visual color
        self.editgraph.edge_artists[edge].set_facecolor(rgb)
        self.canvas.draw()
        #IMPORTANT: update model
        self.design()

    from PyQt5.QtWidgets import QColorDialog
    def pick_edge_color(self):
        if not self.selected_edge:
            QMessageBox.warning(self, "No edge selected", "Select an edge first.")
            return
        color = QColorDialog.getColor()
        if not color.isValid():
            return
        rgb = (color.redF(), color.greenF(), color.blueF())
        self._apply_edge_color(rgb)

    from PyQt5.QtGui import QColor
    def set_edge_color_from_palette(self, color_name):
        if not self.selected_edge:
            QMessageBox.warning(self, "No edge selected", "Select an edge first.")
            return
        color = QColor(color_name)
        rgb = (color.redF(), color.greenF(), color.blueF())
        self._apply_edge_color(rgb)

    def setup_color_menu(self):
        from PyQt5.QtGui import QColor, QIcon, QPixmap

        self.color_menu = QMenu(self)
        colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan"]

        for color_name in colors:
            pixmap = QPixmap(20, 20)
            pixmap.fill(QColor(color_name))

            action = self.color_menu.addAction(QIcon(pixmap), color_name.capitalize())
            action.triggered.connect(
                lambda checked, c=color_name: self.set_family_color(c)
            )
        self.ColorEdgeButton.setMenu(self.color_menu)

    def set_family_color(self, color_name):
        if not self.selected_edge:
            QMessageBox.warning(self, "No edge selected", "Select an edge first.")
            return
        from PyQt5.QtGui import QColor
        color = QColor(color_name)
        rgb = (color.redF(), color.greenF(), color.blueF())

        edge = self.selected_edge
        if not hasattr(self, "edge_family_colors"):
            self.edge_family_colors = {}

        self.edge_family_colors[edge] = rgb
        self._draw_family_circle(edge, rgb)

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

    def _draw_family_circle(self, edge, rgb):
        if self.editgraph is None:
            return
        if edge not in self.editgraph.edge_artists:
            return

        pos = self.editgraph.node_positions
        if edge[0] not in pos or edge[1] not in pos:
            return

        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]

        # midpoint of edge
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        # Offset (so it doesn't sit on the line)
        dx = x2 - x1
        dy = y2 - y1
        norm_val = (dx**2 + dy**2)**0.5 or 1

        offset = 0.015 * self.designsize
        ox = -dy / norm_val * offset
        oy = dx / norm_val * offset

        circle = plt.Circle((x + ox, y + oy), 0.005 * self.designsize, color=rgb, zorder=10)
        
        self.canvas.axes.add_patch(circle)
        self.family_circles[edge]=circle

    def _clear_family_circles(self):
        if hasattr(self, "family_circles"):
            for circle in self.family_circles.values():
                try:
                    circle.remove()
                except Exception:
                    pass
            self.family_circles = {}

    def zoom_in(self):
        self.designsize *= 1.2
        self._apply_zoom()

    def zoom_out(self):
        self.designsize *= 0.8
        self._apply_zoom()

    def reset_zoom(self):
        self.designsize = 2.0
        self._apply_zoom()

    def _apply_zoom(self):
        if self.editgraph is None:
            return

        positions = self.editgraph.node_positions
        edge_color = {edge: self.editgraph.edge_artists[edge].get_facecolor()
                      for edge in self.editgraph.edge_artists}
        modules = {edge: self.editgraph.edge_label_artists[edge].get_text()
                   for edge in self.editgraph.edge_label_artists}
        node_labels = {idt: text.get_text()
                       for idt, text in self.editgraph.node_label_artists.items()}

        g = nx.DiGraph()
        g.add_nodes_from(self.editgraph.nodes)
        g.add_edges_from(self.editgraph.edges)

        self._redraw_graph_with_edges(g, positions, edge_color, modules, node_labels)

    def resizeEvent(self, event, **kwargs):
        """
        Adjusts the current graphical interface parameters and components based on a resize event. It retrieves the
        existing graphical configurations, calculates appropriate scaling values, and reconfigures the graphical
        elements accordingly to maintain visual and functional consistency.
            :param event: Resize event triggering the adjustment process.
            :type event: QResizeEvent
            :param kwargs: Additional keyword arguments that might be required for expansion or specific functionality.
            :type kwargs: dict
            :return: None
        """
        #STEP: Fix the size related to the network design
        width = self.frameGeometry().width()
        height = self.frameGeometry().height()
        self.designsize = min(round(1350 / min(width, height), 1), 2.5)  #heuristic function providing the designsize w.r.t. to window size.

        #STEP: retrieve all parameters of editgraph
        positions = self.editgraph.node_positions
        edge_color = {edge: self.editgraph.edge_artists[edge].get_facecolor() for edge in self.editgraph.edge_artists}
        modules = {edge: self.editgraph.edge_label_artists[edge].get_text() for edge in self.editgraph.edge_label_artists}
        node_labels = {idt: text.get_text() for idt, text in self.editgraph.node_label_artists.items()}

        #STEP: Define a new Editable graph with size updated.
        g = nx.DiGraph()
        g.add_nodes_from(self.editgraph.nodes)
        g.add_edges_from(self.editgraph.edges)

        self.canvas.axes.clear()
        #WARNING : the definition of EditableGraph is the same as the definition in setup_design function. Any modification of one must be reported to the other.
        self.editgraph = EditableGraph(g,
                                       node_labels=node_labels,  #node labels must be explicitly defined to avoid unwanted integer labeling.
                                       node_color='antiquewhite',
                                       node_edge_color='black',
                                       node_label_fontdict=dict(family='sans-serif', color='black', weight='semibold', fontsize=11),
                                       node_layout=positions,
                                       node_size=self.designsize,
                                       node_edge_width=self.designsize / 4,
                                       node_label_offset=(0., 0.025 * self.designsize),
                                       edge_width=self.designsize / 2,
                                       arrows=True,
                                       edge_labels=modules,
                                       edge_label_position=0.75,
                                       edge_label_fontdict=dict(family='sans-serif', fontweight='bold', fontsize=8, color='royalblue'),
                                       edge_color=edge_color,
                                       ax=self.canvas.axes)

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
            self.setup_design()
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
            self.setup_design()
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
            self.boon = self.history[hindex]  #restore the previous BooN
            self.hindex = hindex  #update the history index
            self.refresh()
            self.setup_design()

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
            self.setup_design()

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
        #PRINT THE HISTORY
        #self.show_history()

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
        self.parent.boon.dnf() #convert all formulas into DNF
        self.initialize_view()  #Refresh the view with converted formulas


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
        Updates the current style setting from the user interface and refreshes the stable states view.
            :return: None.
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
        self.moveToThread(self.thread)
        self.thread.start()

    def run(self):
        """
        Defines the run function to execute the primary application logic and signal
        completion. The function encompasses two main operations: invoking the
        application logic and signaling the end of the process.
        """
        self.app()  #run the application
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


#DEF:  MAIN
if __name__ == "__main__":
    app = QApplication(sys.argv)
    boonify = Boonify()
    callback = boonify.canvas.mpl_connect('draw_event', lambda _: boonify.design())
    boonify.show()
    sys.exit(app.exec_())
