IMAGE_NAME=as3438_m1cw
CONTAINER_NAME=as3438_m1cw_container

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run docker container with all the output folders mounted as volumes
run: build
	docker run -it --name $(CONTAINER_NAME) \
	-v "$(PWD)/src/q1/outputs":/usr/src/app/src/q1/outputs \
	-v "$(PWD)/src/q2/outputs":/usr/src/app/src/q2/outputs \
	-v "$(PWD)/src/q3/outputs":/usr/src/app/src/q3/outputs \
	-v "$(PWD)/src/q4/outputs":/usr/src/app/src/q4/outputs \
	-v "$(PWD)/src/q5/outputs":/usr/src/app/src/q5/outputs \
	$(IMAGE_NAME) /bin/bash

# Stop and remove the container
clean:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

# Remove the Docker image
clean-image:
	docker rmi $(IMAGE_NAME)
