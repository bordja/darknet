#include "tcp-ip-client.h"

#include <stdio.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>

#define PORT 27015
#define SERVER_ADDR_INET "192.168.52.2"
#define SA struct sockaddr

struct sockaddr_in servaddr;
struct sockaddr_in cli;
int sockfd = -1;
int connfd;
int len;

int tcp_init_client(void)
{
	// socket create and varification
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd == -1) {
		printf("socket creation failed...\n");
		return -1;
	}
	else
		printf("Socket successfully created..\n");
	bzero(&servaddr, sizeof(servaddr));

	// assign IP, PORT
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr(SERVER_ADDR_INET);
	servaddr.sin_port = htons(PORT);

	// connect the client socket to server socket
	if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0) {
		printf("connection with the server failed...\n");
		return -1;
	}
	else
		printf("connected to the server..\n");
	return 0;
}

void tcp_deinit_client(void)
{
	// close the socket
	close(sockfd);
}
