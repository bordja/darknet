#ifndef __TCP_IP_SERVER_H__
#define __TCP_IP_SERVER_H__

extern int sockfd;

int tcp_init_client(void);
void tcp_deinit_client(void);

#endif /*__TCP_IP_SERVER_H__*/