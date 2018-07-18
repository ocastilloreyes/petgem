#!/usr/bin/env python3
# Author:  Octavio Castillo Reyes
# Contact: octavio.castillo@bsc.es
''' Define functions for mesh handling.
'''


class gmshObject:
    ''' Class for mesh of type Gmsh. This class provides methods
    to parse .msh files to python format.
    '''
    import numpy as np

    def __init__(self, mshfilename):
        ''' Init a gmshObject.

        :param str mshFile: mesh file to be initialized.
        :return: gmshObject initialized.
        '''

        # Read mesh file
        self.mshfilename = mshfilename
        self.mshfID = open(mshfilename, 'r')

        # Init variables
        readingNodes = 0
        readingElements = 0
        self.nNodes = 0
        self.nElems = 0
        self.physical_groups = []
        self.nodes_in_groups = {}
        self.number_physical_groups = 0
        self.nodes_rules = []
        self.elements_rules = []

        linenumber = 1
        # Read file and identify sections whithin it
        for line in self.mshfID:
            # Get beginning of nodes section
            if line.find('$Nodes') >= 0:
                readingNodes = 1
                continue

            # Get beginning of elements section
            if line.find('$Elements') >= 0:
                readingElements = 1
                continue

            # Get end of elements section
            if line.find('$EndElements') >= 0:
                readingElements = 0
                continue

            # Get end of nodes section
            if line.find('$EndNodes') >= 0:
                readingNodes = 0
                continue

            # Read number of nodes (nNodes) if this
            # is the first line of nodes section
            if readingNodes == 1:
                self.nNodes = np.int32(line)
                readingNodes = 2
                continue

            # Read number of elements (nElems) if this
            # is the first line of elements section
            if readingElements == 1:
                self.nElems = np.int32(line)
                readingElements = 2
                continue

            # Sparse elements and populate the list of nodes in groups
            if readingElements == 2:
                current_line = np.array(line.split(), dtype=np.int32)

                # Get properties from header section
                eletag = current_line[0]
                eletype = current_line[1]
                ntags = current_line[2]
                physgrp = 0
                partition = 0

                if ntags >= 2:
                    physgrp = current_line[3]
                    nodelist = current_line[(3 + ntags)::]

                    if physgrp in self.physical_groups:
                        self.nodes_in_groups[physgrp][nodelist] = 1
                    else:
                        self.nodes_in_groups[physgrp] = -np.ones(self.nNodes+1,
                                                                 dtype=np.int16
                                                                 )
                        self.nodes_in_groups[physgrp][nodelist] = 1
                        self.physical_groups.append(physgrp)
                        pass
                else:
                    self.__gmshError__(self.mshfilename + '.msh file has < 2 '
                                       'tags at line ' + str(linenumber))

            linenumber += 1

        # Number of physical groups
        self.number_physical_groups = len(self.physical_groups)
        # Close file
        self.mshfID.close()

    def __gmshPrint__(self, msg):
        ''' Print a message related with a gmshObject.

        :param str msg: message to be printed.
        '''
        print(msg)

        return

    def __gmshError__(self, msg):
        ''' Print error message in a gmshObject and exit.

        :param str msg: message to be printed.
        :return: None.
        '''
        sys.stderr.write('   gmshObject: Error! -> ' + msg + '\n')

    def __destroy__(self):
        ''' Destroy a gmshObject.

        :return: None.
        '''
        self.mshfID.close()

    def printNumberNodes(self):
        ''' Print number of nodes of a gmshObject.

        :return: None.
        '''
        print('   Mesh has ' + str(self.nNodes) + ' nodes')

        return

    def printNumberElements(self):
        ''' Print number of elements of a gmshObject.

        :return: None.
        '''
        print('   Mesh has ' + str(self.nElems) + ' elements')

        return

    def printNumberPhysicalGroups(self):
        ''' Print number of physical groups of a gmshObject.

        :return: None.
        '''
        print('   Mesh has ' + str(self.number_physical_groups) +
              ' physical groups')

        return

    def addElementsRule(self, condition, action):
        ''' Add (append) an user rule to list of elements in the gmshObject.

        :param str condition: function or condition to be added.
        :param str action: action to be executed.
        :return: None.
        '''
        self.elements_rules.append((condition, action))
        pass

    def addNodesRule(self, condition, action):
        ''' Add (append) an user rule to list of nodes in the gmshObject.

        :param str condition: function or condition to be added.
        :param str action: action to be executed.
        :return: None.
        '''
        self.nodes_rules.append((condition, action))
        pass

    def cleanRules(self):
        ''' Clean rules in the gmshObject.
        '''
        self.nodes_rules = []
        self.elements_rules = []
        pass

    def gmshParser(self):
        ''' Parser nodesRules and nodesElements in an gmshObject.
        '''

        # Open Gmsh file
        self.mshfID = open(self.mshfilename, 'r')

        # Move to nodes section
        line = self.mshfID.readline()
        while(line.find('$Nodes') < 0):
            line = self.mshfID.readline()
            pass
        # This line should contain number of nodes (nNodes)
        line = self.mshfID.readline()
        # Verify that nNodes in file is still the nNodes in memory
        if(not np.int32(line) == self.nNodes):
            self.__gmshError__('The number of nodes in ' +
                               self.mshfilename + ' is not consistent.' +
                               ' Aborting')
            exit(-1)

        # Rules for nodes
        if len(self.nodes_rules) == 0:
            # No rules, therefore skip nodes section
            for i in np.arange(self.nNodes):
                self.mshfID.readline()
        else:
            # Read nodes section and apply nodes rules
            for i in np.arange(self.nNodes):
                # Parse the line
                current_line = self.mshfID.readline().split()
                tag = np.int32(current_line[0])
                x = np.double(current_line[1])
                y = np.double(current_line[2])
                z = np.double(current_line[3])

                # Determine the groups to which this node belongs
                physgroups = []
                for grp in self.physical_groups:
                    if self.nodes_in_groups[grp][tag] == 1:
                        physgroups.append(grp)

                for condition, action in self.nodes_rules:
                    if condition(tag, x, y, z, physgroups):
                        action(tag, x, y, z)
                    pass

        # Read another 2 lines after nodes section is done
        # Read $EndNodes line
        line = self.mshfID.readline()
        # Read #Elements line
        line = self.mshfID.readline()
        # Next line should contain number of elements
        line = self.mshfID.readline()

        # Verify that nElems in file is still the nElems in memory
        if(not np.int32(line) == self.nElems):
            self.__gmshError__('The number of elements in ' +
                               self.mshfilename + '.msh is not consistent.' +
                               ' Aborting')
            exit(-1)

        # Rules for elements
        if len(self.elements_rules) == 0:
            # No rules, therefore skip elements section
            for i in np.arange(self.nElems):
                self.mshfID.readline()
        else:
            # Read elements section and apply nodes rules
            nodes = []
            for i in np.arange(self.nElems):
                # Parse the line
                current_line = self.mshfID.readline().split()
                eletag = np.int32(current_line[0])
                eletype = np.int32(current_line[1])
                ntags = np.int32(current_line[2])
                physgrp = np.int32(current_line[3])
                partition = np.int32(current_line[4])

                if ntags >= 2:
                    physgrp = np.int32(current_line[3])
                    nodes = np.array(current_line[(3 + ntags)::],
                                     dtype=np.int32)

                    for condition, action in self.elements_rules:
                        if condition(eletag, eletype, physgrp, nodes):
                            action(eletag, eletype, physgrp, nodes)
                        pass
                else:
                    self.__gmshError__(self.mshfilename + '.msh file has < ' +
                                       '2 tags element with tag ' +
                                       str(elementTag))

        pass

    # Element definitions in Gmsh
    # 2-node line.
    line_2_node = np.int32(1)
    # 3-node triangle.
    triangle_3_node = np.int32(2)
    # 4-node quadrangle
    quadrangle_4_node = np.int32(3)
    # 4-node tetrahedron.
    tetrahedron_4_node = np.int32(4)
    # 8-node hexahedron.
    hexahedron_8_node = np.int32(5)
    # 6-node prism.
    prism_6_node = np.int32(6)
    # 5-node pyramid.
    pyramid_5_node = np.int32(7)
    # 3-node second order line (2 nodes associated with the vertices and
    # 1 with the edge).
    line_3_node = np.int32(8)
    # 6-node second order triangle (3 nodes associated with the vertices
    # and 3 with the edges).
    triangle_6_node = np.int32(9)
    # 9-node second order quadrangle (4 nodes associated with the vertices,
    # 4 with the edges and 1 with the face).
    quadrangle_9_node = np.int32(10)
    # 10-node second order tetrahedron (4 nodes associated with the vertices
    # and 6 with the edges).
    tetrahedron_10_node = np.int32(11)
    # 27-node second order hexahedron (8 nodes associated with the vertices,
    # 12 with the edges, 6 with the faces and 1 with the volume).
    hexahedron_27_node = np.int32(12)
    # 18-node second order prism (6 nodes associated with the vertices,
    # 9 with the edges and 3 with the quadrangular faces).
    prism_18_node = np.int32(13)
    # 14-node second order pyramid (5 nodes associated with the vertices,
    # 8 with the edges and 1 with the quadrangular face).
    pyramid_14_node = np.int32(14)
    # 1-node point.
    point_1_node = np.int32(15)
    # 8-node second order quadrangle (4 nodes associated with the vertices
    # and 4 with the edges).
    quadrangle_8_node = np.int32(16)
    # 20-node second order hexahedron (8 nodes associated with the vertices
    # and 12 with the edges).
    hexahedron_20_node = np.int32(17)
    # 15-node second order prism (6 nodes associated with the vertices
    # and 9 with the edges).
    prism_15_node = np.int32(18)
    # 13-node second order pyramid (5 nodes associated with the vertices
    # and 8 with the edges).
    pyramid_13_node = np.int32(19)
    # 9-node third order incomplete triangle (3 nodes associated with
    # the vertices, 6 with the edges)
    triangle_9_node_incomplete = np.int32(20)
    # 10-node third order triangle (3 nodes associated with the vertices,
    # 6 with the edges, 1 with the face)
    triangle_10_node = np.int32(21)
    # 12-node fourth order incomplete triangle (3 nodes associated with
    # the vertices, 9 with the edges)
    triangle_12_node_incomplete = np.int32(22)
    # 15-node fourth order triangle (3 nodes associated with the vertices,
    # 9 with the edges, 3 with the face)
    triangle_15_node = np.int32(23)
    # 15-node fifth order incomplete triangle (3 nodes associated with
    # the vertices, 12 with the edges)
    triangle_15_node_incomplete = np.int32(24)
    # 21-node fifth order complete triangle (3 nodes associated with the
    # vertices, 12 with the edges, 6 with the face)
    triangle_21_node = np.int32(25)
    # 4-node third order edge (2 nodes associated with the vertices,
    # 2 internal to the edge)
    edge_4_node = np.int32(26)
    # 5-node fourth order edge (2 nodes associated with the vertices,
    # 3 internal to the edge)
    edge_5_node = np.int32(27)
    # 6-node fifth order edge (2 nodes associated with the vertices,
    # 4 internal to the edge)
    edge_6_node = np.int32(28)
    # 20-node third order tetrahedron (4 nodes associated with the vertices,
    # 12 with the edges, 4 with the faces)
    tetrahedron_20_node = np.int32(29)
    # 35-node fourth order tetrahedron (4 nodes associated with the vertices,
    # 18 with the edges, 12 with the faces, 1 in the volume)
    tetrahedron_35_node = np.int32(30)
    # 56-node fifth order tetrahedron (4 nodes associated with the vertices,
    # 24 with the edges, 24 with the faces, 4 in the volume)
    tetrahedron_56_node = np.int32(31)
    # 64-node third order hexahedron (8 nodes associated with the vertices,
    # 24 with the edges, 24 with the faces, 8 in the volume)
    hexahedron_64_node = np.int32(92)
    # 125-node fourth order hexahedron (8 nodes associated with the vertices,
    # 36 with the edges, 54 with the faces, 27 in the volume)
    hexahedron_125_node = np.int32(93)


def getMeshInfo(elemsN, edgesN, bEdges, rank):
    ''' Get and print data mesh.

    :param matrix elemsN: elements-nodes connectivity.
    :param matrix edgesN: edges-nodes connectivity.
    :param vector bEdges: boundary edges.
    :return: number of elements, number of edges, number
             of boundaries and number of degrees of freedom
    :rtype: int
    '''
    # Number of elements
    nElems = elemsN.getSize()[0]
    # Number of edges
    nEdges = edgesN.getSize()[0]
    # Number of boundaries
    nBoundaries = bEdges.getSize()
    # Number of dofs
    ndofs = nEdges - nBoundaries
    # Print mesh information
    if rank == 0:
        PETSc.Sys.Print('\nMesh information')
        PETSc.Sys.Print('='*75)
        PETSc.Sys.Print('  Number of elements:        {0:12}'.
                        format(str(nElems)))
        PETSc.Sys.Print('  Number of edges:           {0:12}'.
                        format(str(nEdges)))
        PETSc.Sys.Print('  Number of dofs:            {0:12}'.
                        format(str(ndofs)))
        PETSc.Sys.Print('  Number of boundary edges:  {0:12}'.
                        format(str(nBoundaries)))

    return nElems, nEdges, nBoundaries, ndofs


def readGmshNodes(mesh_file):
    ''' Read a mesh nodes from a Gmsh file.

    :param str mesh_file: mesh file to be readed.
    :return: nodes coordinates and number of nodes.
    :rtype: ndarray, int.
    '''
    # Create a gmshObject
    inputMesh = gmshObject(mesh_file)

    # Get number of nodes
    nNodes = inputMesh.nNodes

    # Allocate resources
    nodes = np.zeros((nNodes+1, 3), dtype=np.float)

    # Define condition for nodes elemsN, nElems

    def isNode(tag, x, y, z, physgroups):
        ''' Determine if a point is node.

        :param int tag: node ID.
        :param float x: x coordinate of tag.
        :param float y: y coordinate of tag.
        :param float z: z coordinate of tag.
        :param int physgroups: physgroup of tag.
        '''
        return True

    # Define action for nodes
    def getNode(tag, x, y, z):
        ''' Get coordinates of a node point.

        :param int tag: node ID.
        :param float x: x coordinate of tag.
        :param float y: y coordinate of tag.
        :param float z: z coordinate of tag.
        :return: nodal coordinates.
        :rtype: ndarray
        '''
        nodes[tag, :] = [x, y, z]

    # Add nodes rule (composed by a condition and an action)
    inputMesh.addNodesRule(isNode, getNode)

    # Parse mesh file to get nodal coordinates
    inputMesh.gmshParser()

    # Delete first column of nodes since it does not
    # correspond to any node
    nodes = np.delete(nodes, (0), axis=0)

    return nodes, nNodes


def readGmshConnectivity(mesh_file):
    ''' Read a mesh connectivity from a Gmsh file.

    :param str mesh_file: mesh file to be readed.
    :return: mesh connectivity (elemsN) and number of elements.
    :rtype: ndarray, int.
    '''
    # Create a gmshObject
    inputMesh = gmshObject(mesh_file)

    # Get number of nodes
    nElems = inputMesh.nElems

    # Allocate resources
    elemsN = np.zeros((nElems+1, 4), dtype=np.int)

    # Define condition for elements
    def isTetrahedralElement(eletag, eletype, physgrp, nodes):
        ''' Determine if an element is of linear tetrahedral type.

        :param int eletag: element ID.
        :param int eletype: element type (defined by Gmsh).
        :param int physgrp: physgroup of eletag.
        :param ndarray nodes: nodal indexes of eletag.
        :return: element type
        :rtype: int
        '''
        return eletype == inputMesh.tetrahedron_4_node

    # Define action for nodes
    def getElement(eletag, eletype, physgrp, nodes):
        ''' Get element connectivity.

        :param int eletag: element ID.
        :param int eletype: element type (defined by Gmsh).
        :param int physgrp: physgroup of eletag.
        :param ndarray nodes: nodal indexes of eletag.
        :return: element connectivity
        :rtype: ndarray
        '''
        elemsN[eletag, :] = nodes

    # Add nodes rule (composed by a condition and an action)
    inputMesh.addElementsRule(isTetrahedralElement, getElement)

    # Parse mesh file to get nodal coordinates
    inputMesh.gmshParser()

    # Delete first column of connectivity since it does not
    # correspond to any element
    elemsN = np.delete(elemsN, (0), axis=0)
    # Convert to a 0-base numbering in connectivity
    elemsN = elemsN-1

    return elemsN, nElems


def readGmshPhysicalGroups(mesh_file):
    ''' Read conductivity model from a mesh in Gmsh format.

    :param str mesh_file: mesh file to be readed.
    :return: conductivity model (elemsS) and number of elements.
    :rtype: ndarray, int.
    '''
    # Create a gmshObject
    inputMesh = gmshObject(mesh_file)

    # Get number of nodes
    nElems = inputMesh.nElems

    # Allocate resources
    elemsS = np.zeros((nElems+1, 1), dtype=np.int)

    # Define condition for elements
    def isTetrahedralElement(eletag, eletype, physgrp, nodes):
        ''' Determine if an element is of linear tetrahedral type.

        :param int eletag: element ID.
        :param int eletype: element type (defined by Gmsh).
        :param int physgrp: physgroup of eletag.
        :param ndarray nodes: nodal indexes of eletag.
        :return: element type
        :rtype: int
        '''
        return eletype == inputMesh.tetrahedron_4_node

    # Define action for physical groups
    def getPhysicalGroup(eletag, eletype, physgrp, nodes):
        ''' Get element connectivity.

        :param int eletag: element ID.
        :param int eletype: element type (defined by Gmsh).
        :param int physgrp: physgroup of eletag.
        :param ndarray nodes: nodal indexes of eletag.
        :return: element connectivity
        :rtype: ndarray
        '''
        elemsS[eletag] = physgrp

    # Add nodes rule (composed by a condition and an action)
    inputMesh.addElementsRule(isTetrahedralElement, getPhysicalGroup)

    # Parse mesh file to get physical groups for each element
    inputMesh.gmshParser()

    # Delete first column of connectivity since it does not
    # correspond to any element
    elemsS = np.delete(elemsS, (0), axis=0)
    # Convert to a 0-base numbering in the conductivity model
    elemsS = elemsS-1

    return elemsS, nElems


def unitary_test():
    ''' Unitary test for mesh.py script.
    '''

if __name__ == '__main__':
    # Standard module import
    import numpy as np
    import sys
    import petsc4py
    from petsc4py import PETSc
    unitary_test()
else:
    # Standard module import
    import numpy as np
    import sys
    import petsc4py
    from petsc4py import PETSc
