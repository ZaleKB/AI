# -------------------------------------------------------
# Project (1)
# Written by (Junwei Zhang,  40050122)
# For COMP 6721 Section (FK) – Fall 2019
# --------------------------------------------------------


import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import time

class GeometryMap:

    def __init__(self):
        self.mapp_to_Use = None  #map represent each block if it is over threshold or not
        self.mapp = None        #map represent total crime rate on each block
        self.city = gpd.read_file('E:\COMP 6721\shapefile\crime_dt.shp')
        self.coord = self.city['geometry']

        self.indexing = pd.DataFrame([self.coord.x, self.coord.y], index=['x', 'y']).T
        self.city_coord = np.array(self.indexing)  #turn coordination of each points to matrix 

        self.difference = 0.04       #the area of the map
        self.length_grid = 0.002

        self.grid_side_number = math.ceil(0.04 / self.length_grid)  

        self.end_x = -73.59 + (self.length_grid * self.grid_side_number)
        self.end_y = round((45.49 + (self.length_grid * self.grid_side_number)), 6)

    def show_map(self):
        self.city.plot(cmap='jet', column='CATEGORIE', figsize=(9, 9))  

        plt.savefig("OriginalMap.png")

    #create grid on map depends on size of grid and count crime rate inside each block
    def create_grid(self):
        
        plt.figure(figsize=(9, 9))

        x = np.linspace(-73.59, self.end_x, self.grid_side_number + 1)
        y = np.linspace(45.49, self.end_y, self.grid_side_number + 1)

        global xx
        global yy
        xx, yy = np.meshgrid(x, y, indexing='xy')


        matrix_Count = np.zeros((self.grid_side_number, self.grid_side_number))  # block_side_count

        rows = len(matrix_Count)
        cols = len(matrix_Count[0])

        #count how many points locate on each block
        for i in range(len(self.city_coord)):
            for j in range(cols):
                if self.city_coord[i][0] >= xx[0][j] and self.city_coord[i][0] <= xx[0][j + 1]:
                    locate_x = j
                    break

            for k in range(rows):
                if self.city_coord[i][1] >= yy[k][0] and self.city_coord[i][1] <= yy[k + 1][0]:
                    locate_y = k
                    break

            matrix_Count[locate_x][locate_y] += 1

        self.mapp = matrix_Count.copy()
        
    
        plt.plot(xx, yy, 'k-', lw=0.5, color='k')
        plt.plot(xx.T, yy.T, 'k-', lw=0.5, color='k')
        

    #show information that needed
    def display_information(self):
        print('The average crime rates for totall blocks is', self.mapp.mean())
        print('---------------------------------------------------------------')
        print('The standard deviation of crimes for totall blocks is', np.std(self.mapp))
        print('---------------------------------------------------------------')

        for i in range(len(self.mapp)):
            for j in range(len(self.mapp[0])):
                print('The total number of crimes on block row',
                      i, 'column', j, 'is', self.mapp[i][j])

    #paint color separate block using threshold and generate matrix used later for A* algorithm
    def paint_block(self, threshold):

        matrix_Use = np.zeros((self.grid_side_number, self.grid_side_number))


        average_number = self.mapp.mean()
        
        value = average_number * (threshold - 0.5 + 1)

        for i in range(self.grid_side_number):
            for j in range(self.grid_side_number):
                if self.mapp[i][j] >= value:
                    matrix_Use[i][j] = 1

        self.mapp_to_Use = np.zeros((len(matrix_Use), len(matrix_Use[0])))
        
        for n in range(len(matrix_Use)):
            self.mapp_to_Use[n] = matrix_Use[len(matrix_Use) - 1 - n]
        
        
        plt.figure(figsize=(9, 9))

        plt.pcolor(xx, yy, matrix_Use, linewidths=1) # paint color, separate block by 0, 1
        
        return self.mapp_to_Use
        
        

''' we design our A star algorithm '''
class Point:
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Node:


    isHardPath = False  # check if a certain path is between a yellow and blue block (1.3 cost)

    def __init__(self, point, g=0, h=0):
        self.point = point  # coordination for this node
        self.father = None  # father node
        self.g = g  # g value
        self.h = h  # h value

    """
    h(n) function, use manhattan function
     """

    def manhattan(self, endNode):
        self.h = (abs(endNode.point.x - self.point.x) + abs(endNode.point.y - self.point.y))

    def setG(self, g):
        self.g = g

    def setFather(self, node):
        self.father = node


class AStar:
    """
    A* Algorithm logic
    """
    #self.data = None

    def __init__(self, map2d, startNode, endNode):
        """
        map2d:      looking for map to print
        startNode:  looking for startNode
        endNode:    looking for endnode
        """
        self.data = np.zeros((3,4))
        
        self.openList = []

        self.closeList = []
        # map data
        self.map2d = map2d

        self.startNode = startNode

        self.endNode = endNode

        self.currentNode = startNode
        # last generated path
        self.pathlist = [];
        return

    def getMinFNode(self):
        """
        get min_value in openlist
        """
        nodeTemp = self.openList[0]
        for node in self.openList:
            if node.g + node.h < nodeTemp.g + nodeTemp.h:
                nodeTemp = node
        return nodeTemp

    def nodeInOpenlist(self, node):    #check if a node is in openlist
        for nodeTmp in self.openList:
            if nodeTmp.point.x == node.point.x \
                    and nodeTmp.point.y == node.point.y:
                return True
        return False

    def nodeInCloselist(self, node):   #check if a node is in closelist
        for nodeTmp in self.closeList:
            if nodeTmp.point.x == node.point.x \
                    and nodeTmp.point.y == node.point.y:
                return True
        return False

    def endNodeInOpenList(self):
        for nodeTmp in self.openList:
            if nodeTmp.point.x == self.endNode.point.x \
                    and nodeTmp.point.y == self.endNode.point.y:
                return True
        return False

    def getNodeFromOpenList(self, node):
        for nodeTmp in self.openList:
            if nodeTmp.point.x == node.point.x \
                    and nodeTmp.point.y == node.point.y:
                return nodeTmp
        return None

    def searchOneNode(self, node):
        """
        搜索一个节点
        x为是行坐标
        y为是列坐标
        """

        # ignore the barrier
        if self.map2d.isPass(node.point) != True:
            return
            # ignore the close list
        if self.nodeInCloselist(node):
            return
            # G-value calculate
        if abs(node.point.x - self.currentNode.point.x) == 1 and abs(node.point.y - self.currentNode.point.y) == 1:
            gTemp = 1.5
        elif node.isHardPath:
            gTemp = 1.3
        else:
            gTemp = 1.0

            # if not in openlist then put into openlist
        if self.nodeInOpenlist(node) == False:
            node.setG(self.currentNode.g + gTemp)
            # H calculation
            node.manhattan(self.endNode);
            self.openList.append(node)
            node.father = self.currentNode
        # if inside openlist，check if g-value frome currentNode to This node is smaller
        # if it is, then recalculate g-value and change its father
        else:
            nodeTmp = self.getNodeFromOpenList(node)
            if self.currentNode.g + gTemp < nodeTmp.g:
                nodeTmp.g = self.currentNode.g + gTemp
                nodeTmp.father = self.currentNode
        return;


    def isInsideAndWork(self, x, y):   #check if a node's point out of bounds
        try:
            if (x <= len(self.data) -1 and x >= 0) and (y >= 0 and y <= len(self.data[0] - 1) and self.data[x][y] == 1):
                return True
            else:
                return False
        except IndexError:
            return False


    def searchNear(self):
        """
        search near 8 points
        position is as follow
        (x-1,y-1)(x-1,y)(x-1,y+1)
        (x  ,y-1)(x  ,y)(x  ,y+1)
        (x+1,y-1)(x+1,y)(x+1,y+1)
        """
        
        """check if there are 3 yellow blocks around a point"""
        if (self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y - 1)) and \
                (self.isInsideAndWork(self.currentNode.point.x - 1,self.currentNode.point.y)) and \
                (self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y)):
            left = Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1))
            down = Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y))
            down.isHardPath = True
            left.isHardPath = True
            self.searchOneNode(down)
            self.searchOneNode(left)
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y - 1)))

        elif (self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y - 1)) and \
                (self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y-1)) and \
                (self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y)):
            up = Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y))
            right = Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1))
            up.isHardPath = True
            right.isHardPath = True
            self.searchOneNode(up)
            self.searchOneNode(right)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))

        elif (self.isInsideAndWork(self.currentNode.point.x - 1,self.currentNode.point.y)) and \
                (self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y-1)) and \
                (self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y)):
            up = Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y))
            left = Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1))
            up.isHardPath = True
            left.isHardPath = True
            self.searchOneNode(up)
            self.searchOneNode(left)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))

        elif (self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y - 1)) and \
                (self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y-1)) and \
                (self.isInsideAndWork(self.currentNode.point.x - 1,self.currentNode.point.y)):
            down = Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y))
            right = Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1))
            down.isHardPath = True
            right.isHardPath = True
            self.searchOneNode(down)
            self.searchOneNode(right)
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y + 1)))


        elif (self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y - 1)) and \
                (self.isInsideAndWork(self.currentNode.point.x - 1,self.currentNode.point.y)):
            
            left = Node(Point(self.currentNode.point.x, self.currentNode.point.y-1))
            right = Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1))
            left.isHardPath = True
            right.isHardPath = True
            self.searchOneNode(left)
            self.searchOneNode(right)
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y-1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y+1)))


        elif (self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y - 1)) and \
                self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y-1):
            up = Node(Point(self.currentNode.point.x-1, self.currentNode.point.y))
            down = Node(Point(self.currentNode.point.x+1, self.currentNode.point.y))
            up.isHardPath = True
            down.isHardPath = True
            self.searchOneNode(up)
            self.searchOneNode(down)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y+1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y + 1)))

        elif self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y-1) and \
                self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y):
            left = Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1))
            right = Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1))
            left.isHardPath = True
            right.isHardPath = True
            self.searchOneNode(left)
            self.searchOneNode(right)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))

        elif (self.isInsideAndWork(self.currentNode.point.x - 1,self.currentNode.point.y)) and \
                self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y):
            up = Node(Point(self.currentNode.point.x-1, self.currentNode.point.y))
            down = Node(Point(self.currentNode.point.x+1, self.currentNode.point.y))
            up.isHardPath = True
            down.isHardPath = True
            self.searchOneNode(up)
            self.searchOneNode(down)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y-1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y - 1)))

        elif (self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y - 1)) and \
                self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y):
            up = Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y))
            down = Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y))
            left = Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1))
            right = Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1))
            up.isHardPath = True
            down.isHardPath = True
            left.isHardPath = True
            right.isHardPath = True
            self.searchOneNode(up)
            self.searchOneNode(down)
            self.searchOneNode(left)
            self.searchOneNode(right)
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))

        elif (self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y-1)) and \
                self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y):
            up = Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y))
            down = Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y))
            left = Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1))
            right = Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1))
            up.isHardPath = True
            down.isHardPath = True
            left.isHardPath = True
            right.isHardPath = True
            self.searchOneNode(up)
            self.searchOneNode(down)
            self.searchOneNode(left)
            self.searchOneNode(right)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y + 1)))

        
        elif self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y - 1):
            up = Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y))
            left = Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1))
            up.isHardPath = True
            left.isHardPath = True
            self.searchOneNode(up)
            self.searchOneNode(left)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y - 1)))

        elif self.isInsideAndWork(self.currentNode.point.x - 1, self.currentNode.point.y):
            up = Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y))
            right = Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1))
            up.isHardPath = True
            right.isHardPath = True
            self.searchOneNode(up)
            self.searchOneNode(right)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y + 1)))

        elif self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y):
            down = Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y))
            right = Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1))
            down.isHardPath = True
            right.isHardPath = True
            self.searchOneNode(down)
            self.searchOneNode(right)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y - 1)))

        elif self.isInsideAndWork(self.currentNode.point.x, self.currentNode.point.y-1):
            down = Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y))
            left = Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1))
            down.isHardPath = True
            left.isHardPath = True
            self.searchOneNode(down)
            self.searchOneNode(left)
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y + 1)))


        else:
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y)))
            self.searchOneNode(Node(Point(self.currentNode.point.x - 1, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y + 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y)))
            self.searchOneNode(Node(Point(self.currentNode.point.x + 1, self.currentNode.point.y - 1)))
            self.searchOneNode(Node(Point(self.currentNode.point.x, self.currentNode.point.y - 1)))

        return;


    def start(self):
        '''''
        begin searching
        '''
        # put startNode into openlist
        self.startNode.manhattan(self.endNode);
        self.startNode.setG(0);
        self.openList.append(self.startNode)

        while True:
            # fetch min_F value node from openlist
            # put it into closelist and delete it in openlist
            self.currentNode = self.getMinFNode()
            self.closeList.append(self.currentNode)
            self.openList.remove(self.currentNode)

            self.searchNear()


            # check if the endNode find it and put all node in pathlist all the way through the beginning
            if self.endNodeInOpenList():
                nodeTmp = self.getNodeFromOpenList(self.endNode)
                while True:
                    self.pathlist.append(nodeTmp);
                    if nodeTmp.father != None:
                        nodeTmp = nodeTmp.father
                    else:
                        return True;
            elif len(self.openList) == 0:
                return False;
        return True;
    
    def setMap(self):
        for node in self.pathlist:
            self.map2d.setMap(node.point);
        return;


class map2d:
    """
    map data
    """

    def __init__(self, side_n):
        self.side_n = side_n
        
        self.grid_web = np.ones((self.side_n + 1, self.side_n + 1))   # longitude and latitude coordinate

        self.pathTag = 8

           
    def setMap(self, point):
        self.grid_web[point.x][point.y] = self.pathTag
        return

    #check is a point is in-bound
    def isPass(self, point):

        if (point.x <= 0 or point.x >= self.side_n) or (point.y <= 0 or point.y >= self.side_n):
            return False;
        else:
            return True


'''draw path line on grid map'''       
class draw:
    
    def __init__(self, web, size):
        self.web = web   #path shows from matrix
        self.size = size  #each block size
        self.maxy = 45.53  #we pinpoint startpoint as (-73.59, 45.53)
        self.minx = -73.59

    def plot_line(self):
        plt.figure(figsize=(9, 9))

        x_list = []
        y_list = []

        for i in range(len(self.web[0])):
            for j in range(len(self.web)):
                if self.web[j][i] == 8:
                    x_coor = self.minx + i * self.size
                    y_coor = self.maxy - j * self.size
                    x_list.append(x_coor)
                    y_list.append(y_coor)
             
        
        plt.plot(xx, yy, 'k-', lw=0.5, color='k')
        plt.plot(xx.T, yy.T, 'k-', lw=0.5, color='k')
        plt.plot(x_list, y_list, color='r', lw=1.0)


if __name__ == '__main__':
    ##construct the map
    mapToShow = GeometryMap()
    mapToShow.show_map()
    mapToShow.create_grid()
    mapToShow.display_information()
    distribution = mapToShow.paint_block(0.5)  # use for check barrier
    
    mapTest = map2d(mapToShow.grid_side_number)
    side_n = mapToShow.grid_side_number
    #mapTest.side_n = mapToShow.grid_side_number
    #mapTest.showMap()
    ##build A* algorithm
    aStar = AStar(mapTest, Node(Point(1,1)), Node(Point(side_n - 1,side_n - 1)))
    aStar.data = distribution
    print ("A* start:")
    ##start looking path
    if aStar.start():
        aStar.setMap();
        #mapTest.showMap();
    else:
        print ("Due to blocks, no path is found. Please change the map and try again")

    #print(mapTest.grid_web)
    start = time.clock()
    pathMap = draw(mapTest.grid_web,mapToShow.length_grid) #path shows
    pathMap.plot_line()
    elapsed = (time.clock() - start)
    print('Time used to generate path is:', elapsed)