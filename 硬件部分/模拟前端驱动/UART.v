//Distance_2024/1/7
module UART(
clk,
rst_n,
tx_uart,
data_in,
//en_uart,
start,
AFE_RDY              //开始传输（拨码）
);

parameter   sys_clk = 50000000;
parameter   bps = 115200;
parameter   number_of_bytes = 6;//一次发几个字节


input          clk;
input          rst_n;
input[number_of_bytes*8-1:0]    data_in; 
//input          en_uart;
input          start;
input          AFE_RDY;
output reg     tx_uart;



(* preserve *)reg[number_of_bytes*8-1:0]      data_in_eff;
reg[7:0]     din_uart;    //一次发送的八位数据
reg         en_send;
//reg         en_uart_ff0;
//reg         en_uart_ff1;
//reg         en_uart_ff2;
wire        add_cnt;
wire        end_cnt;
(* preserve *)reg[12:0]   cnt;         //记录位宽
wire        add_cnt1;
(* keep *)wire        end_cnt1;
(* preserve *)reg[3:0]    cnt1;        //记录位数
wire[9:0]   data_out;
reg [3:0]   cnt_bytes;   //记录发送了几个字节
reg         data_busy;
reg         byte_done;
(* preserve *)reg[2:0]    state;
reg         AFE_RDY_eff;
(* preserve *)reg[number_of_bytes*8-1:0]  data_in_buf;

localparam  capture = 3'b000;
localparam  buffer_renew = 3'b001; 
localparam  eff_renew = 3'b010; 


(* preserve *)reg[2:0]     state_cap;
//捕捉AFE信号
always @( posedge clk or negedge rst_n  )begin

   if(!rst_n)begin
   AFE_RDY_eff <= 0;
   data_in_buf<= 0;
	data_in_eff<= 0;
   state_cap<= capture;
   end
   else begin
   
     case(state_cap)
          capture:
   		 begin
			 AFE_RDY_eff <= 0;
   		     if(!rst_n)begin
              AFE_RDY_eff <= 0;
   		     data_in_buf<= 0;
   			  state_cap<= capture;
   	//		en_send <= 0;
   	//		data_busy <=0 ;
              end
              else if(AFE_RDY)begin
   			  state_cap<= buffer_renew;
				  
   	        end
   
   		 end
   		 
			
          buffer_renew:
   		 begin
   		 data_in_buf <= data_in;
   		    if(data_busy==0)begin
             state_cap<= eff_renew;
   		    end
   			 else if(data_busy==1)begin
   		    state_cap<= buffer_renew;
   		    end
   		 end
   		 
   		 eff_renew:
   		 begin
   		 
   		 data_in_eff <= data_in_buf;
			 AFE_RDY_eff <=1;
   		 state_cap<= capture;
   		 end
   		
   		 default: state_cap <= capture;
      endcase 
   	
   end
end
	

  localparam IDLE = 3'b000;
  localparam STATE1 = 3'b001;
  localparam STATE2 = 3'b010;
  localparam STATE3 = 3'b011;
  localparam STATE4 = 3'b100;
  localparam STATE5 = 3'b101;
  localparam STATE6 = 3'b110;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)begin
      state <= IDLE;
	//	byte_done<=0;
	 en_send <= 0;
	 data_busy <=0 ;
	 end
    else begin
      case (state)
        IDLE: begin
          if (start ==1 && AFE_RDY_eff ==1)begin
            state <= STATE1;
            data_busy <=1;				
			//	byte_done==0;
			 end
          else begin
            state <= IDLE;
				data_busy <=0;
	//		   byte_done<=0; 
          end				// 保持在当前状态
        end
		  
        STATE1: 
		  begin
		     if(end_cnt1==0)begin
           din_uart <= data_in_eff[7:0];
		     state <= STATE1;
		     en_send <= 1;
		     end
		     else begin
		     state <= STATE2;
		     en_send <= 0;
	//		  byte_done <= 0;
		     end
        end
		  
        STATE2: 
		  begin
		     if(end_cnt1==0)begin
           din_uart <= data_in_eff[15:8];
		     state <= STATE2;
		     en_send <= 1;
		     end
		     else begin
		     state <= STATE3;
		     en_send <= 0;
	//		  byte_done <= 0;
		     end
        end
		  
        STATE3: 
		  begin
		     if(end_cnt1==0)begin
           din_uart <= data_in_eff[23:16];
		     state <= STATE3;
		     en_send <= 1;
		     end
		     else begin
		     state <= STATE4;
		     en_send <= 0;
	//		  byte_done <= 0;
		     end
        end
		  
        STATE4: 
		  begin
		     if(end_cnt1==0)begin
           din_uart <= data_in_eff[31:24];
		     state <= STATE4;
		     en_send <= 1;
		     end
		     else begin
		     state <= STATE5;
		     en_send <= 0;
	//		  byte_done <= 0;
		     end
        end
		  
        STATE5: 
		  begin
		     if(end_cnt1==0)begin
           din_uart <= data_in_eff[39:32];
		     state <= STATE5;
		     en_send <= 1;
		     end
		     else begin
		     state <= STATE6;
		     en_send <= 0;
		//	  byte_done <= 0;
		     end
        end
		  
        STATE6: 
		  begin
		     if(end_cnt1==0)begin
           din_uart <= data_in_eff[47:40];
		     state <= STATE6;
		     en_send <= 1;
		     end
		     else begin
		     state <= IDLE;
			  data_busy <=0;
		     en_send <= 0;
	//		  byte_done <= 0;
		     end
        end
		  
		  
        default: state <= IDLE; // 默认情况下返回到空闲状态
      endcase
    end
  end

always @(posedge clk or negedge en_send)begin
    if(en_send==0)begin
        cnt <= 0;
	//	  byte_done <=0;
    end
    else if(add_cnt && end_cnt1 == 0 )begin
        if(end_cnt /*|| en_send ==0*/)
            cnt <= 0;
        else
            cnt <= cnt + 1;
	//			byte_done <=0;
    end
end

assign add_cnt = 1 ;       
assign end_cnt = add_cnt && cnt== sys_clk/bps -1;   


always @(posedge clk or negedge en_send)begin
    if(en_send==0)begin
        cnt1 <= 0;
		  byte_done <=0;
    end
	 
    else if(add_cnt1||end_cnt1)begin
        if(end_cnt1)begin
            cnt1 <= 0;
	//			byte_done <=1;
			//	kk<=1;
      //      tx_uart <= 1; 
		   //   en_send <= 0;		
        end				//一个字节发送完成
        else
            cnt1 <= cnt1 + 1;
    end
end

assign add_cnt1 = end_cnt ;       
assign end_cnt1 =  cnt1== 10 ; 
assign data_out = { 1'b1,din_uart[7:0],1'b0};

always  @(posedge clk or negedge en_send)begin
    if(en_send==0)begin                   //如果en_send信号为0 串口将挂起
        tx_uart <= 1;
    end
	 else if(start == 0) begin
        tx_uart <= 1;
    end
    else if(cnt1 == 0) begin
        tx_uart <= 0;
    end
 else if(/*end_cnt &&*/ cnt1<10)begin
        tx_uart <= data_out[cnt1];
    end
end

endmodule

