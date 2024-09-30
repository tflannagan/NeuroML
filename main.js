// ML Entity Simulation with Generational Learning
let selectedEntity = null;
let neuronActivations = [];
// Constants
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;
const MODEL_CANVAS_WIDTH = 400;
const MODEL_CANVAS_HEIGHT = 300;
const POPULATION_SIZE = 20;
const INITIAL_FOOD_COUNT = 300;
const MAX_FOOD_COUNT = 300;
const FOOD_SPAWN_RATE = 0.05;
const GENERATION_DURATION = 1000;

const WORLD_WIDTH = 3000;
const WORLD_HEIGHT = 800;
const VIEW_WIDTH = 600;
const VIEW_HEIGHT = 600;
let viewX = 0;
let viewY = 0;
const ZOOM_LEVELS = [0.5, 0.75, 1, 1.25, 1.5];
let currentZoomIndex = 2;

// Color scheme
const COLORS = {
    background: '#0a0a0a',
    grid: '#1a1a1a',
    food: '#00ffff',
    entity: '#ff00ff',
    neuronPositive: '#00ffff',
    neuronNegative: '#ff00ff',
    text: '#00ffff'
};

if (!tf.getBackend() || tf.getBackend() !== 'webgl') {
    console.warn('WebGL not supported. Falling back to CPU backend.');
    tf.setBackend('cpu');
}

// Neural Network
class NeuralNetwork {
    constructor(inputNodes, hiddenLayers, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenLayers = hiddenLayers;
        this.outputNodes = outputNodes;

        const layers = [
            tf.layers.dense({ inputShape: [inputNodes], units: hiddenLayers[0], activation: 'relu' }),
            ...hiddenLayers.slice(1).map(units => tf.layers.dense({ units, activation: 'relu' })),
            tf.layers.dense({ units: outputNodes, activation: 'tanh' })
        ];

        this.model = tf.sequential({ layers });
        this.model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
    }

    predict(inputs) {
        return this.model.predict(tf.tensor2d([inputs])).dataSync();
    }

    copy() {
        const copyNet = new NeuralNetwork(this.inputNodes, this.hiddenLayers, this.outputNodes);
        const weights = this.model.getWeights();
        const weightCopies = weights.map(w => w.clone());
        copyNet.model.setWeights(weightCopies);
        return copyNet;
    }

    mutate(rate) {
        const weights = this.model.getWeights();
        const mutatedWeights = weights.map(w => {
            let tensor = w.clone();
            let values = tensor.dataSync().slice();
            for (let i = 0; i < values.length; i++) {
                if (Math.random() < rate) {
                    values[i] += randomGaussian() * 0.1;
                }
            }
            return tf.tensor(values, tensor.shape);
        });
        this.model.setWeights(mutatedWeights);
    }
}


// Entity
class Entity {
        constructor(x, y, brain = null) {
            this.x = Math.min(Math.max(x, 0), WORLD_WIDTH);
            this.y = Math.min(Math.max(y, 0), WORLD_HEIGHT);
        this.direction = Math.random() * Math.PI * 2;
        this.speed = 0;
        this.energy = 100;
        this.fitness = 0;
        this.age = 0;
        this.lastMealDistance = 0;
        this.brain = brain || new NeuralNetwork(10, [16, 16, 8], 3);
        this.neuronActivations = [];
        this.color = this.generateColor();
        this.size = 20;
        this.lastPosition = { x: this.x, y: this.y };
        this.stagnantTicks = 0;
        this.movementThreshold = 0.5;
        }
    
        generateColor() {
            return `hsl(${Math.random() * 360}, 100%, 50%)`;
        }
    

        update(foods, entities, obstacles) {
            const closestFood = this.findClosestFood(foods);
            const closestEntity = this.findClosestEntity(entities);
            const closestObstacle = this.findClosestObstacle(obstacles);
            const inputs = [
                this.x / WORLD_WIDTH,
                this.y / WORLD_HEIGHT,
                this.energy / 100,
                this.age / GENERATION_DURATION,
                closestFood ? (closestFood.x - this.x) / WORLD_WIDTH : 0,
                closestFood ? (closestFood.y - this.y) / WORLD_HEIGHT : 0,
                closestEntity ? (closestEntity.x - this.x) / WORLD_WIDTH : 0,
                closestEntity ? (closestEntity.y - this.y) / WORLD_HEIGHT : 0,
                closestObstacle ? (closestObstacle.x - this.x) / WORLD_WIDTH : 0,
                closestObstacle ? (closestObstacle.y - this.y) / WORLD_HEIGHT : 0
            ];
    
            this.neuronActivations = this.forwardPass(inputs);
            const [direction, speed, reproduce] = this.brain.predict(inputs);
            
            this.direction = direction * Math.PI * 2;
            this.speed = (speed + 1) * 2;
            
            let newX = this.x + Math.cos(this.direction) * this.speed;
            let newY = this.y + Math.sin(this.direction) * this.speed;
            
            // Apply border collision
            if (newX < 0) {
                newX = 0;
                this.direction = Math.PI - this.direction;
            } else if (newX > WORLD_WIDTH) {
                newX = WORLD_WIDTH;
                this.direction = Math.PI - this.direction;
            }
            
            if (newY < 0) {
                newY = 0;
                this.direction = -this.direction;
            } else if (newY > WORLD_HEIGHT) {
                newY = WORLD_HEIGHT;
                this.direction = -this.direction;
            }
    
            
            // Normalize direction to [0, 2π]
            this.direction = (this.direction + 2 * Math.PI) % (2 * Math.PI);
            
            // Check for collisions with obstacles
            if (!this.collidesWithObstacles(newX, newY, obstacles)) {
                this.x = newX;
                this.y = newY;
            } else {
                // Bounce off obstacle
                this.direction = Math.random() * Math.PI * 2;
                this.energy -= 5; // Penalty for hitting an obstacle
            }

            // Check for stagnant behavior
        const distanceMoved = Math.hypot(this.x - this.lastPosition.x, this.y - this.lastPosition.y);
        if (distanceMoved < this.movementThreshold) {
            this.stagnantTicks++;
            if (this.stagnantTicks > 50) { // Adjust this threshold as needed
                this.energy -= 0.5; // Penalty for being stagnant
                this.fitness -= 0.1; // Reduce fitness for stagnant behavior
            }
        } else {
            this.stagnantTicks = 0;
        }

        // Update last position
        this.lastPosition = { x: this.x, y: this.y };
            
            this.energy -= 0.1 + (this.speed * 0.05);
            this.fitness += 0.1;
            this.age += 1;
            this.size = Math.max(5, Math.min(15, this.energy / 10));
    
            if (reproduce > 0.8 && this.energy > 50) {
                this.reproduce();
            }
        }
    
        collidesWithObstacles(x, y, obstacles) {
            return obstacles.some(obstacle => obstacle.contains(x, y));
        }
    
        findClosestObstacle(obstacles) {
            return obstacles.reduce((closest, obstacle) => {
                const distance = Math.hypot(obstacle.x - this.x, obstacle.y - this.y);
                return (!closest || distance < closest.distance) ? { ...obstacle, distance } : closest;
            }, null);
        }

    draw(ctx) {
        ctx.save();
        ctx.translate(this.x, this.y);
        ctx.rotate(this.direction);

        ctx.fillStyle = COLORS.entity;
        ctx.strokeStyle = COLORS.entity;
        ctx.lineWidth = 2;

        const size = Math.max(5, Math.min(15, this.energy / 10));

        switch (this.shape) {
            case 0: // Circle
                ctx.beginPath();
                ctx.arc(0, 0, size, 0, Math.PI * 2);
                ctx.fill();
                break;
            case 1: // Triangle
                ctx.beginPath();
                ctx.moveTo(size, 0);
                ctx.lineTo(-size, -size);
                ctx.lineTo(-size, size);
                ctx.closePath();
                ctx.fill();
                break;
            case 2: // Square
                ctx.fillRect(-size, -size, size * 2, size * 2);
                break;
        }

        // Draw direction indicator
        ctx.beginPath();
        ctx.moveTo(size, 0);
        ctx.lineTo(size + 5, 0);
        ctx.stroke();

        ctx.restore();
    }


    forwardPass(inputs) {
        let activations = [inputs];
        let currentInput = tf.tensor2d([inputs]);

        this.brain.model.layers.forEach(layer => {
            const output = layer.apply(currentInput);
            activations.push(Array.from(output.dataSync()));
            currentInput = output;
        });

        return activations;
    }

    findClosestFood(foods) {
        return foods.reduce((closest, food) => {
            const distance = Math.hypot(food.x - this.x, food.y - this.y);
            if (!closest || distance < closest.distance) {
                this.lastMealDistance = distance;
                return { ...food, distance };
            }
            return closest;
        }, null);
    }

    findClosestEntity(entities) {
        return entities.reduce((closest, entity) => {
            if (entity === this) return closest;
            const distance = Math.hypot(entity.x - this.x, entity.y - this.y);
            return (!closest || distance < closest.distance) ? { ...entity, distance } : closest;
        }, null);
    }

    reproduce() {
        if (entities.length < POPULATION_SIZE * 1.5) {
            const childBrain = this.brain.copy();
            childBrain.mutate(mutationRate);
            const child = new Entity(this.x, this.y, childBrain);
            entities.push(child);
            this.energy -= 30;
        }
    }
}

// Food

class Food {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 3 + 2;
    }

    draw(ctx) {
        ctx.fillStyle = COLORS.food;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

class Obstacle {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 5 + 5; // Random size between 5 and 10 (smaller than before)
        this.vertices = this.generateVertices();
        this.color = this.generateColor();
    }

    generateVertices() {
        const numVertices = Math.floor(Math.random() * 3) + 5; // 5 to 7 vertices
        const vertices = [];
        for (let i = 0; i < numVertices; i++) {
            const angle = (i / numVertices) * Math.PI * 2;
            const distance = this.size * (0.8 + Math.random() * 0.4); // 80% to 120% of size
            const x = Math.cos(angle) * distance;
            const y = Math.sin(angle) * distance;
            vertices.push({ x, y });
        }
        return vertices;
    }

    generateColor() {
        const baseColor = [0, 255, 255]; // Cyan base color
        const variation = 30; // Color variation range
        return `rgb(${baseColor[0] + Math.random() * variation - variation / 2},
                    ${baseColor[1] + Math.random() * variation - variation / 2},
                    ${baseColor[2] + Math.random() * variation - variation / 2})`;
    }

    draw(ctx) {
        ctx.fillStyle = this.color;
        ctx.strokeStyle = '#ff00ff'; // Tron pink outline
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        ctx.moveTo(this.x + this.vertices[0].x, this.y + this.vertices[0].y);
        for (let i = 1; i < this.vertices.length; i++) {
            ctx.lineTo(this.x + this.vertices[i].x, this.y + this.vertices[i].y);
        }
        ctx.closePath();
        
        ctx.fill();
        ctx.stroke();
    }

    contains(x, y) {
        // Check if a point is inside the obstacle (for collision detection)
        let inside = false;
        for (let i = 0, j = this.vertices.length - 1; i < this.vertices.length; j = i++) {
            const xi = this.x + this.vertices[i].x, yi = this.y + this.vertices[i].y;
            const xj = this.x + this.vertices[j].x, yj = this.y + this.vertices[j].y;
            const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }
}


// Simulation
let entities = [];
let foods = [];
let isRunning = false;
let generation = 0;
let ticks = 0;
let timeScale = 1;
let simulationSpeed = 1;
let mutationRate = 0.1;
let obstacles = [];

function initSimulation() {
    entities = Array.from({ length: POPULATION_SIZE }, () => 
        new Entity(Math.random() * WORLD_WIDTH, Math.random() * WORLD_HEIGHT)
    );
    foods = Array.from({ length: INITIAL_FOOD_COUNT }, () => 
        new Food(Math.random() * WORLD_WIDTH, Math.random() * WORLD_HEIGHT)
    );
    obstacles = Array.from({ length: 10 }, () => 
        new Obstacle(Math.random() * WORLD_WIDTH, Math.random() * WORLD_HEIGHT)
    );
    
    ticks = 0;
    updateStats();
    updateSelectedEntity();
}

function updateSimulation() {
    if (!isRunning) return;
    
    for (let i = 0; i < simulationSpeed; i++) {
        entities.forEach(entity => {
            entity.update(foods, entities, obstacles);
            
            const eatenFood = foods.findIndex(food => 
                Math.hypot(food.x - entity.x, food.y - entity.y) < entity.size + food.size
            );
            
            if (eatenFood !== -1) {
                entity.energy += foods[eatenFood].size * 10;
                entity.fitness += 10 + (100 - entity.lastMealDistance) / 10;
                foods.splice(eatenFood, 1);
            }
        });
        
        // Spawn new food
        if (foods.length < MAX_FOOD_COUNT && Math.random() < FOOD_SPAWN_RATE) {
            foods.push(new Food(Math.random() * WORLD_WIDTH, Math.random() * WORLD_HEIGHT));
        }
        
        // Remove dead entities
        entities = entities.filter(entity => entity.energy > 0);
        
        ticks++;
        
        if (ticks >= GENERATION_DURATION || entities.length === 0) {
            nextGeneration();
        }
    }
    
    updateSelectedEntity();
    drawSimulation();
    drawModel();
    updateStats();
    requestAnimationFrame(updateSimulation);
}
function nextGeneration() {
    if (entities.length === 0) {
        initSimulation();
        return;
    }
    generation++; 
    ticks = 0;
    
    // Calculate fitness
    const totalFitness = entities.reduce((sum, entity) => sum + entity.fitness, 0);
    
    // Create new population
    const newPopulation = [];
    while (newPopulation.length < POPULATION_SIZE) {
        const parent = selectParent(totalFitness);
        const child = new Entity(
            Math.random() * WORLD_WIDTH,
            Math.random() * WORLD_HEIGHT,
            parent.brain.copy()
        );
        child.brain.mutate(mutationRate);
        newPopulation.push(child);
    }
    
    entities = newPopulation;

    entities.forEach(entity => {
        entity.fitness = 0;
        entity.energy = 100;
    });

    // Spawn new food
    while (foods.length < INITIAL_FOOD_COUNT) {
        foods.push(new Food(Math.random() * WORLD_WIDTH, Math.random() * WORLD_HEIGHT));
    }

    updateStats(); 
}


function selectParent(totalFitness) {
    let runningSum = 0;
    const threshold = Math.random() * totalFitness;
    for (let entity of entities) {
        runningSum += entity.fitness;
        if (runningSum > threshold) {
            return entity;
        }
    }
    return entities[entities.length - 1];
}

function drawSimulation() {
    const canvas = document.getElementById('simulationCanvas');
    const ctx = canvas.getContext('2d');
    
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, VIEW_WIDTH, VIEW_HEIGHT);
    
    const zoom = ZOOM_LEVELS[currentZoomIndex];
    ctx.save();
    ctx.scale(zoom, zoom);
    ctx.translate(-viewX, -viewY);

    // Draw grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1 / zoom;
    for (let i = 0; i < WORLD_WIDTH; i += 50) {
        ctx.beginPath();
        ctx.moveTo(i, 0);
        ctx.lineTo(i, WORLD_HEIGHT);
        ctx.stroke();
    }
    for (let i = 0; i < WORLD_HEIGHT; i += 50) {
        ctx.beginPath();
        ctx.moveTo(0, i);
        ctx.lineTo(WORLD_WIDTH, i);
        ctx.stroke();
    }
    
    foods.forEach(food => food.draw(ctx));
    obstacles.forEach(obstacle => obstacle.draw(ctx));
    entities.forEach(entity => entity.draw(ctx));
    
    if (selectedEntity) {
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 2 / zoom;
        ctx.beginPath();
        ctx.arc(selectedEntity.x, selectedEntity.y, selectedEntity.size + 5, 0, Math.PI * 2);
        ctx.stroke();
    }

    ctx.restore();

    // Draw navigation controls
    drawNavigationControls(ctx);
}


function drawModel() {
    const canvas = document.getElementById('modelCanvas');
    const ctx = canvas.getContext('2d');
    
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;
    
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    
    if (!selectedEntity) {
        updateSelectedEntity();
    }
    
    if (!selectedEntity) return;
    
    const weights = selectedEntity.brain.model.getWeights();
    const layers = [10, 16, 16, 8, 3];
    const padding = WIDTH * 0.05; // Reduce padding to 2% of width
    const layerWidth = (WIDTH - 2 * padding) / (layers.length - 1);
    
    const fontSize = Math.max(8, Math.floor(HEIGHT / 50)); // Adjust font size based on height
    ctx.font = `${fontSize}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < layers.length; i++) {
        const neurons = layers[i];
        const x = padding + i * layerWidth;
        const neuronSpacing = (HEIGHT - 2 * padding) / (neurons + 1);
        
        for (let j = 0; j < neurons; j++) {
            const y = padding + (j + 1) * neuronSpacing;
            
            const activation = selectedEntity.neuronActivations[i] ? selectedEntity.neuronActivations[i][j] || 0 : 0;
            ctx.fillStyle = activation > 0 ? COLORS.neuronPositive : COLORS.neuronNegative;
            ctx.globalAlpha = Math.abs(activation);
            ctx.beginPath();
            ctx.arc(x, y, fontSize / 2, 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1;
            
            // Display activation value
            ctx.fillStyle = COLORS.text;
            ctx.fillText(activation.toFixed(2), x, y + fontSize);
            
            if (i < layers.length - 1) {
                const nextNeurons = layers[i + 1];
                const nextNeuronSpacing = (HEIGHT - 2 * padding) / (nextNeurons + 1);
                const layerWeights = weights[i * 2].arraySync();
                
                for (let k = 0; k < nextNeurons; k++) {
                    const nextY = padding + (k + 1) * nextNeuronSpacing;
                    const weight = layerWeights[j] ? layerWeights[j][k] : 0;
                    const weightStrength = Math.abs(weight);
                    ctx.strokeStyle = weight > 0 ? COLORS.neuronPositive : COLORS.neuronNegative;
                    ctx.lineWidth = weightStrength * fontSize / 10;
                    ctx.globalAlpha = weightStrength;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(x + layerWidth, nextY);
                    ctx.stroke();
                    ctx.globalAlpha = 1;

                    // Only display weight values for the last layer to reduce clutter
                    if (i === layers.length - 2) {
                        const midX = (x + x + layerWidth) / 2;
                        const midY = (y + nextY) / 2;
                        ctx.fillStyle = COLORS.text;
                        ctx.fillText(weight.toFixed(2), midX, midY);
                    }
                }
            }
        }
    }

    // Display layer types
    const layerTypes = ['Input', 'Hidden', 'Hidden', 'Hidden', 'Output'];
    ctx.fillStyle = COLORS.text;
    ctx.font = `bold ${fontSize * 1.2}px Arial`;
    for (let i = 0; i < layers.length; i++) {
        const x = padding + i * layerWidth;
        ctx.fillText(layerTypes[i], x, padding / 2);
    }
}

function updateStats() {
    const avgFitness = entities.length > 0 ? entities.reduce((sum, entity) => sum + entity.fitness, 0) / entities.length : 0;
    const bestFitness = entities.length > 0 ? Math.max(...entities.map(entity => entity.fitness)) : 0;
    document.getElementById('stats').innerHTML = `
        <span style="color: ${COLORS.text}">
            Gen: ${generation} | Avg Fitness: ${avgFitness.toFixed(2)} | Best: ${bestFitness.toFixed(2)} | Alive: ${entities.length} | Food: ${foods.length}
        </span>
    `;
}

function updateSelectedEntity() {
    selectedEntity = entities.length > 0 ? entities.reduce((best, current) => 
        (current.fitness > best.fitness) ? current : best
    ) : null;
}

function drawNavigationControls(ctx) {
    const padding = 10;
    const buttonSize = 30;
    const fontSize = 20;

    ctx.fillStyle = COLORS.text;
    ctx.font = `${fontSize}px Arial`;

    // Zoom level
    ctx.fillText(`Zoom: ${ZOOM_LEVELS[currentZoomIndex]}x`, padding, VIEW_HEIGHT - padding);

    // Navigation buttons
    const buttons = [
        { text: '←', x: VIEW_WIDTH - 3 * buttonSize - 2 * padding, y: VIEW_HEIGHT - buttonSize - padding },
        { text: '→', x: VIEW_WIDTH - buttonSize - padding, y: VIEW_HEIGHT - buttonSize - padding },
        { text: '↑', x: VIEW_WIDTH - 2 * buttonSize - padding, y: VIEW_HEIGHT - 2 * buttonSize - padding },
        { text: '↓', x: VIEW_WIDTH - 2 * buttonSize - padding, y: VIEW_HEIGHT - buttonSize - padding },
        { text: '+', x: padding, y: VIEW_HEIGHT - buttonSize - padding },
        { text: '-', x: buttonSize + padding, y: VIEW_HEIGHT - buttonSize - padding },
    ];

    buttons.forEach(button => {
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(button.x, button.y, buttonSize, buttonSize);
        ctx.strokeStyle = COLORS.text;
        ctx.strokeRect(button.x, button.y, buttonSize, buttonSize);
        ctx.fillStyle = COLORS.text;
        ctx.fillText(button.text, button.x + buttonSize / 2 - fontSize / 4, button.y + buttonSize / 2 + fontSize / 4);
    });
}

function handleNavigation(event) {
    const rect = event.target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const buttonSize = 30;
    const padding = 10;

    if (y > VIEW_HEIGHT - buttonSize - padding) {
        if (x < 2 * buttonSize + padding) {
            // Zoom buttons
            if (x < buttonSize + padding) {
                zoomIn();
            } else {
                zoomOut();
            }
        } else if (x > VIEW_WIDTH - 3 * buttonSize - 2 * padding) {
            // Left/Right buttons
            if (x < VIEW_WIDTH - 2 * buttonSize - padding) {
                panView(-50, 0);
            } else if (x > VIEW_WIDTH - buttonSize - padding) {
                panView(50, 0);
            } else {
                panView(0, 50);
            }
        }
    } else if (y > VIEW_HEIGHT - 2 * buttonSize - padding && 
               x > VIEW_WIDTH - 2 * buttonSize - padding && 
               x < VIEW_WIDTH - buttonSize - padding) {
        // Up button
        panView(0, -50);
    }
}

function zoomIn() {
    if (currentZoomIndex < ZOOM_LEVELS.length - 1) {
        currentZoomIndex++;
        adjustViewForZoom();
    }
}

function zoomOut() {
    if (currentZoomIndex > 0) {
        currentZoomIndex--;
        adjustViewForZoom();
    }
}

function adjustViewForZoom() {
    const zoom = ZOOM_LEVELS[currentZoomIndex];
    viewX = Math.min(Math.max(0, viewX + VIEW_WIDTH / 2 * (1 - 1 / zoom)), WORLD_WIDTH - VIEW_WIDTH / zoom);
    viewY = Math.min(Math.max(0, viewY + VIEW_HEIGHT / 2 * (1 - 1 / zoom)), WORLD_HEIGHT - VIEW_HEIGHT / zoom);
}

function panView(dx, dy) {
    const zoom = ZOOM_LEVELS[currentZoomIndex];
    viewX = Math.min(Math.max(0, viewX + dx / zoom), WORLD_WIDTH - VIEW_WIDTH / zoom);
    viewY = Math.min(Math.max(0, viewY + dy / zoom), WORLD_HEIGHT - VIEW_HEIGHT / zoom);
}



// Event Listeners

document.getElementById('simulationCanvas').addEventListener('click', handleNavigation);

document.addEventListener('keydown', (event) => {
    switch(event.key) {
        case ' ':
            document.getElementById('startStopButton').click();
            break;
        case 'r':
            document.getElementById('resetButton').click();
            break;
        case 'e':
            document.getElementById('evolveButton').click();
            break;
        case 'a':
            panView(-50, 0);
            break;
        case 'd':
            panView(50, 0);
            break;
        case 'w':
            panView(0, -50);
            break;
        case 'sa':
            panView(0, 50);
            break;
        case '+':
            zoomIn();
            break;
        case '-':
            zoomOut();
            break;
    }
});
document.getElementById('startStopButton').addEventListener('click', () => {
    isRunning = !isRunning;
    document.getElementById('startStopButton').textContent = isRunning ? 'Stop' : 'Start';
    if (isRunning) updateSimulation();
});

document.getElementById('resetButton').addEventListener('click', () => {
    isRunning = false;
    document.getElementById('startStopButton').textContent = 'Start';
    initSimulation();
    drawSimulation();
    drawModel();
});

document.getElementById('evolveButton').addEventListener('click', () => {
    if (!isRunning) {
        nextGeneration();
        drawSimulation();
        drawModel();
        updateStats();
    }
});

document.getElementById('simulationSpeed').addEventListener('input', (event) => {
    simulationSpeed = parseInt(event.target.value);
    document.getElementById('simulationSpeedValue').textContent = `${simulationSpeed}x`;
});

document.getElementById('mutationRate').addEventListener('input', (event) => {
    mutationRate = parseFloat(event.target.value);
    document.getElementById('mutationRateValue').textContent = mutationRate.toFixed(2);
});

document.head.insertAdjacentHTML('beforeend', `
    <style>
        body { background-color: #000; color: ${COLORS.text}; }
        button, input[type="range"] { background-color: #1a1a1a; color: ${COLORS.text}; border: 1px solid ${COLORS.text}; }
        button:hover { background-color: #2a2a2a; }
        #modelCanvas { display: block; margin: 0 auto; }
    </style>
`);

// Initialize
initSimulation();
drawSimulation();
drawModel();
updateStats();

// Handle visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        isRunning = false;
        document.getElementById('startStopButton').textContent = 'Start';
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (event) => {
    switch(event.key) {
        case ' ':
            document.getElementById('startStopButton').click();
            break;
        case 'r':
            document.getElementById('resetButton').click();
            break;
        case 'e':
            document.getElementById('evolveButton').click();
            break;
    }
});

// Utility functions
function randomGaussian() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function getActivationColor(value) {
    const normalizedValue = (value + 1) / 2; // Map [-1, 1] to [0, 1]
    return `rgb(${Math.floor(255 * (1 - normalizedValue))}, ${Math.floor(255 * normalizedValue)}, 0)`;
}

// Initialize
initSimulation();
drawSimulation();
drawModel();
updateStats();

// Add this to handle browser tab visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        isRunning = false;
        document.getElementById('startStopButton').textContent = 'Start Simulation';
    }
});



