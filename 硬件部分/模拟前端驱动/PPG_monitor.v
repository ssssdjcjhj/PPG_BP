//Distance_2024/1/8
module PPG_monitor(


input	wire clk,
input	wire rst_n,
input wire SPISOMI,
input wire AFE_RDY,


output wire SPISIMO,
output wire SPISTE,   
output wire sclk,     

output wire pdn,
  
output wire reset,  
 
output vcc33,
output vcc5,

output wire    test, 
//UART
input       start, 
output wire tx_uart




);

 wire signed [47:0] spi_data_out;

(* keep *) wire[47:0]  spi_data_out_test; 
 
assign  spi_data_out_test = spi_data_out;

//assign test = spi_data_out[0];
assign vcc33 = 1;
assign vcc5 = 1;
//assign spi_data_out = 48'b001000000000000000000000000000000000000000011101;
//parameter reco = 24'b010101000011001000010000
//assign defined_data = {}
spi spi_u1(
        .clk(clk),
        .rst_n(rst_n),
        .SPISOMI(SPISOMI),
        .AFE_RDY(AFE_RDY),
		  
		  
        .SPISIMO(SPISIMO),
        .SPISTE(SPISTE),   
        .sclk(sclk),     
        .data_out(spi_data_out),
        .pdn(pdn),
  
        .reset(reset)      
);

UART UART_u1(

       .clk(clk),
       .rst_n(rst_n),
       .tx_uart(tx_uart),
       .data_in(spi_data_out),
       .start(start),
       .AFE_RDY(AFE_RDY)
              //开始传输（拨码）
);


endmodule
