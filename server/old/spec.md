API specs
===

Reminder: all APIs should be car specific for supporting multi car events

# Cars
GET /cars


# Tubs

GET /tubs/:car
List all tubs available
* path (or ID if somehow possible .. would need a DB)
* metadata

GET /tubs/:car/:id NO NEED??

GET /tubs/:car/:id/stream
-Testbench like video

# Models

GET /models/:car

# Training

POST /train
-Start/Stop

GET /train
-Status

# Deploy

POST /deploy/:car

# Control
