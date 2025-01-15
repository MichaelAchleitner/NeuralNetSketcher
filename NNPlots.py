import matplotlib.pyplot as plt
import numpy as np

def create_neural_network(layer_sizes, layer_labels, max_display_nodes=5):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    ax.set_axis_off()
    
    vertical_distance = 2
    horizontal_distance = 2
    node_radius = 0.2
    layer_positions = []
    
    # Calculate visible nodes for each layer
    visible_nodes = []
    for size in layer_sizes:
        if size <= max_display_nodes:
            visible_nodes.append(list(range(size)))
        else:
            visible_nodes.append([0, 1, 2, -2, -1])
    
    # Position nodes
    for i, layer_nodes in enumerate(visible_nodes):
        layer_pos = []
        actual_size = len(layer_nodes)
        
        for j, node_idx in enumerate(layer_nodes):
            x = i * horizontal_distance
            if actual_size == 5:
                if j < 3:
                    y = vertical_distance - j * vertical_distance/2
                elif j == 3:
                    y = -vertical_distance/2  # Middle dots position
                else:
                    y = -vertical_distance
            else:
                y = (actual_size - 1) * vertical_distance/2 - j * vertical_distance
            layer_pos.append((x, y))
        layer_positions.append(layer_pos)
    
            # Draw edges
        for i in range(len(layer_positions)-1):
            for j, node1 in enumerate(layer_positions[i]):
                for k, node2 in enumerate(layer_positions[i+1]):
                    # Draw lines from visible neurons
                    if j != 3 and k != 3:
                        dx = node2[0] - node1[0]
                        dy = node2[1] - node1[1]
                        dist = np.sqrt(dx**2 + dy**2)
                        dx, dy = dx/dist, dy/dist
                        
                        # Calculate points at circle edges
                        start_x = node1[0] + node_radius * dx
                        start_y = node1[1] + node_radius * dy
                        end_x = node2[0] - node_radius * dx
                        end_y = node2[1] - node_radius * dy
                        
                        plt.plot([start_x, end_x], [start_y, end_y], 'gray', linewidth=0.5)
                    
                    # Draw lines from/to middle dots
                    elif j == 3 or k == 3:
                        dots_y = layer_positions[i if j == 3 else i+1][3][1] - 0.2
                        if j == 3 and k == 3:  # Dots to dots connection
                            plt.plot([node1[0], node2[0]], [dots_y, dots_y],
                                'gray', linewidth=0.5)
                        elif j == 3:  # From dots to next layer neurons
                            dx = node2[0] - node1[0]
                            dy = node2[1] - dots_y
                            dist = np.sqrt(dx**2 + dy**2)
                            dx, dy = dx/dist, dy/dist
                            end_x = node2[0] - node_radius * dx
                            end_y = node2[1] - node_radius * dy
                            plt.plot([node1[0], end_x], [dots_y, end_y],
                                'gray', linewidth=0.5)
                        else:  # From neurons to dots
                            dx = node2[0] - node1[0]
                            dy = dots_y - node1[1]
                            dist = np.sqrt(dx**2 + dy**2)
                            dx, dy = dx/dist, dy/dist
                            start_x = node1[0] + node_radius * dx
                            start_y = node1[1] + node_radius * dy
                            plt.plot([start_x, node2[0]], [start_y, dots_y],
                                'gray', linewidth=0.5)


    
    # Draw nodes and dots
    for i, layer_pos in enumerate(layer_positions):
        for j, (x, y) in enumerate(layer_pos):
            if len(layer_pos) == 5 and j == 3:
                plt.plot([x], [y], 'k.', markersize=5)
                plt.plot([x], [y-0.2], 'k.', markersize=5)
                plt.plot([x], [y-0.4], 'k.', markersize=5)
            elif j != 3:
                circle = plt.Circle((x, y), node_radius, color='white', ec='black')
                ax.add_artist(circle)
    
    # Add layer labels
    for i in range(len(layer_sizes)):
        x = i * horizontal_distance
        y = vertical_distance + 0.8
        plt.text(x, y, layer_labels[i] + " (" + str(layer_sizes[i]) + ")", ha='center')
    
    # Add vertical dotted lines
    for i in range(len(layer_sizes)-1):
        x = i * horizontal_distance + horizontal_distance/2
        plt.vlines(x, -2.5, 2.5, linestyles='dotted', colors='gray')
    
    # Add labels
    for i, (x, y) in enumerate(layer_positions[0]):
        if i < 3:
            plt.text(x-0.5, y, f'Input {i+1}', ha='right')
        elif i == len(layer_positions[0])-1:
            plt.text(x-0.5, y, f'Input {layer_sizes[0]}', ha='right')
    
    for i, (x, y) in enumerate(layer_positions[-1]):
        if i < 3:
            plt.text(x+0.5, y, f'Output {i+1}', ha='left')
        elif i == len(layer_positions[-1])-1:
            plt.text(x+0.5, y, f'Output {layer_sizes[-1]}', ha='left')
    
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Example usage
layer_sizes = [8, 512, 128,1]
layer_labels = ['i', 'h₁', 'h₂', 'o']

create_neural_network(layer_sizes, layer_labels)
