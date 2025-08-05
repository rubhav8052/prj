
Task: Traffic Lights Inlay Classification

You are given an image with a green bounding box that perfectly fits a traffic light inlay. Your goal is to:

Classify the inlay type of the Traffic Light inside the image.

Rules for Inlay classification:

The inlay of a Traffic Light is of type 'Circle' if the glowing sign is a full circle of any colour.
The inlay of a Traffic Light is of type 'ShapeRing' if the glowing sign shows a ring or hollow circle.
The inlay of a Traffic Light is of type 'ShapeX' if the glowing sign shows an X shape.
The inlay of a Traffic Light is of type 'ShapeCross' if the glowing sign looks like a "+" symbol and not X shape.
The inlay of a Traffic Light is of type 'ShapeBar' if the glowing sign shows a rectangular bar in any direction.
The inlay of a Traffic Light is of type 'ShapeDot' if the glowing sign shows an dot shape.
The inlay of a Traffic Light is of type 'ShapeSquare' if the glowing sign shows an square shape.
The inlay of a Traffic Light is of type 'ShapeTriangle' if the glowing sign shows an triangle shape.
The inlay of a Traffic Light is of type 'ShapeBusTram' if the glowing sign shows a bus or a tram.
The inlay of a Traffic Light is of type 'ArrowStraight' if the glowing sign is a single arrow of any colour and is pointing up.
The inlay of a Traffic Light is of type 'ArrowStraightLeft' if the glowing sign is has two arrows, one pointing up and the other left.
The inlay of a Traffic Light is of type 'ArrowStraightRight' if the glowing sign is has two arrows, one pointing up and the other right.
The inlay of a Traffic Light is of type 'ArrowDown' if the glowing sign is a single arrow of any colour and is pointing down.
The inlay of a Traffic Light is of type 'ArrowDownLeft' if the glowing sign is has two arrows, one pointing down and the other left.
The inlay of a Traffic Light is of type 'ArrowDownRight' if the glowing sign is has two arrows, one pointing down and the other right.
The inlay of a Traffic Light is of type 'ArrowLeft' if the glowing sign is an arrow of any colour and is pointing left.
The inlay of a Traffic Light is of type 'ArrowLeft45Up' if the glowing sign is an arrow diagonally facing up and left.
The inlay of a Traffic Light is of type 'ArrowLeft45Down' if the glowing sign is an arrow diagonally facing down and left.
The inlay of a Traffic Light is of type 'ArrowRight' if the glowing sign is an arrow of any colour and is pointing right.
The inlay of a Traffic Light is of type 'ArrowRight45Up' if the glowing sign is an arrow diagonally facing up and right.
The inlay of a Traffic Light is of type 'ArrowRight45Down' if the glowing sign is an arrow diagonally facing down and right.
The inlay of a Traffic Light is of type 'ArrowLeftRight' if the glowing sign has two arrows, one pointing left and the other right.
The inlay of a Traffic Light is of type 'ArrowUTurnLeft' if the glowing sign is an arrow of the shape of an inverted "U" and the head is towards the left.
The inlay of a Traffic Light is of type 'ArrowUTurnRight' if the glowing sign is an arrow of the shape of an inverted "U" and the head is towards the right.
The inlay of a Traffic Light is of type 'ArrowOther' if the glowing sign is an arrow but none of the above.
The inlay of a Traffic Light is of type 'Pedestrian' if the glowing sign depicts a person.
The inlay of a Traffic Light is of type 'Bicycle' if the glowing sign depicts a bicycle.
The inlay of a Traffic Light is of type 'PedestrianBicycle' if the glowing sign shows both a person and a bicycle.
The inlay of a Traffic Light is of type 'CountdownGraphical' if the glowing sign has a graphical representation.
The inlay of a Traffic Light is of type 'CountdownNumerical' if the glowing sign is a number.
The inlay of a Traffic Light is of type 'TextBusOrB' if the glowing sign says "Bus" or just a "B".
The inlay of a Traffic Light is of type 'TextOther' if the glowing sign has some alpha numeric text but cannot be classified as "TextBusOrB".
The inlay of a Traffic Light is of type 'NoTurn' if the sign shows a striking line over a bend arrow, prohibiting any kind of turning of vehicles. 
The inlay of a Traffic Light is of type 'Hand' if the glowing sign shows a human hand.
The inlay of a Traffic Light is of type 'Unsure' if the glowing sign is not clear.
The inlay of a Traffic Light is of type 'Other' you cannot classify with a high confidence.


Output Format:

{"Inlay": "value"}
Replace value with the classified inlay type.
Ensure that the output generated belongs to the list of possible values provided in the Rules.