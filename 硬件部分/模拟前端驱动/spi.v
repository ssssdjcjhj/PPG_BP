module spi(
input	wire clk,
input	wire rst_n,
input wire SPISOMI,
input wire AFE_RDY,


output reg SPISIMO,
output reg SPISTE,   
output reg sclk,     
output reg signed [47:0]data_out,
output reg pdn,
  
output reg reset      
);

//reg [3:0]cnt_tmp = 4'b0 ;
reg sclk_tmp = 1'b0 ;
reg [2:0]state = 3'b0;
reg [27:0]cnt_initial =28'd0;       // d50_000_000 的时钟，使复位1s，见p14,在设计过程中将其写成50000，以便验证
reg [2:0]cnt_sclk = 3'b0 ;          //sclk频率规定不超过16mhz，此处使用5mhz
reg [5:0]cnt_write = 6'b1 ;
reg [5:0]cnt_read = 6'b1 ;
reg [7:0]cnt_data_write = 8'b0 ;    //共41个寄存器需要写
reg [31:0]data_tmp = 32'b0 ;
reg [15:0]cnt_interval = 16'b0 ;      //1ms的时间从写切换到读
reg [7:0]cnt_address_read = 8'b0 ;
reg [7:0]addr_read_tmp = 8'b0 ;
reg [23:0]data_read_tmp = 24'b0 ;
reg [23:0]LED2_substracted = 24'b0 ;
reg [23:0]LED1_substracted = 24'b0 ;
reg [4:0]cnt_write_times = 5'd0 ;


wire AFE_RDY_N;

assign AFE_RDY_N = ~AFE_RDY;

/*
//1KHZ采样率下只有4000个点
parameter W_register_00 = 32'h00000000 ;//在写的时候

parameter W_register_01 = 32'h01000BD1 ;//3025
parameter W_register_02 = 32'h02000F9E ;//3998
parameter W_register_03 = 32'h03000BB8 ;//3000
parameter W_register_04 = 32'h04000F9F ;//3999
parameter W_register_05 = 32'h05000019 ;//25
parameter W_register_06 = 32'h060003E6 ;//998
parameter W_register_07 = 32'h07000401 ;//1025
parameter W_register_08 = 32'h080007CE ;//1998
parameter W_register_09 = 32'h090003E8 ;//1000
parameter W_register_0A = 32'h0A0007CF ;//1999
parameter W_register_0B = 32'h0B0007E9 ;//2025
parameter W_register_0C = 32'h0C000BB6 ;//2998
parameter W_register_0D = 32'h0D000002 ;//2
parameter W_register_0E = 32'h0E0003E7 ;//999
parameter W_register_0F = 32'h0F0003EA ;//1002

parameter W_register_10 = 32'h100007CF ;//1999
parameter W_register_11 = 32'h110007D2 ;//2002
parameter W_register_12 = 32'h12000BB7 ;//2999
parameter W_register_13 = 32'h13000BBA ;//3002
parameter W_register_14 = 32'h14000F9F ;//3999
parameter W_register_15 = 32'h15000000 ;//0
parameter W_register_16 = 32'h16000002 ;//2
parameter W_register_17 = 32'h170003E8 ;//1000
parameter W_register_18 = 32'h180003EA ;//1002
parameter W_register_19 = 32'h190007D0 ;//2000
parameter W_register_1A = 32'h1A0007D2 ;//2002
parameter W_register_1B = 32'h1B000BB8 ;//3000
parameter W_register_1C = 32'h1C000BBA ;//3002
parameter W_register_1D = 32'h1D000F9F ;//3999
                                        //以上是计时器分配  
   */                                    
   /*                                    
parameter W_register_1E = 32'h1E000E02 ;//配置alarm pin用作监控用途，计数器的控制
parameter W_register_1F = 32'h1F000000 ;//用作未来使用
parameter W_register_20 = 32'h20000000 ;//用作生产使用
parameter W_register_21 = 32'h21000000 ;//在所示位数控制放大器的增益、环境变量消除电流和滤波（此处先采用默认）
parameter W_register_22 = 32'h2201FFFF ;//控制两个LED的开关及电流强度$$$
parameter W_register_23 = 32'h23020100 ;//控制LED驱动是全桥或推挽；控制数字输出的三态模式；控制晶振使能；输入输出和AFE的开关
parameter W_register_24 = 32'h24000000 ;//用作未来使用
parameter W_register_25 = 32'h25000000 ;//用作未来使用
parameter W_register_26 = 32'h26000000 ;//用作未来使用
parameter W_register_27 = 32'h27555555 ;//用作生产使用
parameter W_register_28 = 32'h28AAAAAA ;//用作生产使用
parameter W_register_29 = 32'h29000080 ;//控制alarm pin的使能
    */       

		  
//学姐张的
parameter W_register_00 = 32'h00000000 ;//在写的时候

parameter W_register_01 = 32'h010017A2 ;
parameter W_register_02 = 32'h02001F3E ;
parameter W_register_03 = 32'h03001770 ;
parameter W_register_04 = 32'h04001F3F ;
parameter W_register_05 = 32'h05000032 ;
parameter W_register_06 = 32'h060007CE ;
parameter W_register_07 = 32'h07000802 ;
parameter W_register_08 = 32'h08000F9E ;
parameter W_register_09 = 32'h090007D0 ;
parameter W_register_0A = 32'h0A000F9F ;
parameter W_register_0B = 32'h0B000FD2 ;
parameter W_register_0C = 32'h0C00176E ;
parameter W_register_0D = 32'h0D000004 ;
parameter W_register_0E = 32'h0E0007CF ;
parameter W_register_0F = 32'h0F0007D4 ;

parameter W_register_10 = 32'h10000F9F ;
parameter W_register_11 = 32'h11000FA4 ;
parameter W_register_12 = 32'h1200176F ;
parameter W_register_13 = 32'h13001774 ;
parameter W_register_14 = 32'h14001F3F ;
parameter W_register_15 = 32'h15000000 ;
parameter W_register_16 = 32'h16000003 ;
parameter W_register_17 = 32'h170007D0 ;
parameter W_register_18 = 32'h180007D3 ;
parameter W_register_19 = 32'h19000FA0 ;
parameter W_register_1A = 32'h1A000FA3 ;
parameter W_register_1B = 32'h1B001770 ;
parameter W_register_1C = 32'h1C001773 ;
parameter W_register_1D = 32'h1D001F3F ;
                                       //以上是计时器分配  
            
   /*        
parameter W_register_1E = 32'h1E000B02 ;//配置alarm pin用作监控用途，计数器的控制
parameter W_register_1F = 32'h1F000000 ;//用作未来使用
parameter W_register_20 = 32'h20000000 ;//用作生产使用
parameter W_register_21 = 32'h21000000 ;//在所示位数控制放大器的增益、环境变量消除电流和滤波（此处先采用默认）
parameter W_register_22 = 32'h22014020 ;//控制两个LED的开关及电流强度$$$
parameter W_register_23 = 32'h23020100 ;//控制LED驱动是全桥或推挽；控制数字输出的三态模式；控制晶振使能；输入输出和AFE的开关
parameter W_register_24 = 32'h24000000 ;//用作未来使用
parameter W_register_25 = 32'h25000000 ;//用作未来使用
parameter W_register_26 = 32'h26000000 ;//用作未来使用
parameter W_register_27 = 32'h27555555 ;//用作生产使用
parameter W_register_28 = 32'h28AAAAAA ;//用作生产使用
parameter W_register_29 = 32'h29000000 ;//控制alarm pin的使能													 
	*/											 

           
parameter W_register_1E = 32'h1E000B02 ;//配置alarm pin用作监控用途，计数器的控制
parameter W_register_1F = 32'h1F000000 ;//用作未来使用
parameter W_register_20 = 32'h20000000 ;//用作生产使用
parameter W_register_21 = 32'h21004400 ;//在所示位数控制放大器的增益、环境变量消除电流和滤波（此处先采用默认）
//parameter W_register_21 = 32'h21000000 ;
parameter W_register_22 = 32'h2201FFFF ;//控制两个LED的开关及电流强度$$$
parameter W_register_23 = 32'h23020100 ;//控制LED驱动是全桥或推挽；控制数字输出的三态模式；控制晶振使能；输入输出和AFE的开关
parameter W_register_24 = 32'h24000000 ;//用作未来使用
parameter W_register_25 = 32'h25000000 ;//用作未来使用
parameter W_register_26 = 32'h26000000 ;//用作未来使用
parameter W_register_27 = 32'h27555555 ;//用作生产使用
parameter W_register_28 = 32'h28AAAAAA ;//用作生产使用
parameter W_register_29 = 32'h29000000 ;//控制alarm pin的使能													 
												 

/*
state:
000:复位不工作
001：开始前复位
010：开始写入
011:发送间歇
100:发送完成,暂停1ms
101:开始读数据
110:读取间隔

*/

//initial pdn = 1'b1 ;



//状态机设置
//参数为50_000_000
always@ (posedge clk or negedge rst_n)
	begin
		if(rst_n == 1'b0)
			begin
			state <= 3'b0 ;
            cnt_initial <= 0;
            SPISTE <= 1'b1 ;
            pdn <= 1'b0;
            reset <= 1'b0;
				
			end
		else if(cnt_initial < 28'd50_000)
			begin
			state  <= 3'b001 ;
			cnt_initial <= cnt_initial + 1'b1 ;
            SPISTE <= 1'b1 ;
            pdn <= 1'b0;
            reset <= 1'b0;
			end	
        else if((cnt_initial < 28'd10_000_000)&&(cnt_initial >= 28'd50_000))
			begin
			state  <= 3'b001 ;
			cnt_initial <= cnt_initial + 1'b1 ;
            SPISTE <= 1'b1 ;
            pdn <= 1'b1;
            reset <= 1'b0;
			end	
        else if((cnt_initial < 28'd50_000_000)&&(cnt_initial >= 28'd10_000_000))
			begin
			state  <= 3'b001 ;
			cnt_initial <= cnt_initial + 1'b1 ;
            SPISTE <= 1'b1 ;
            pdn <= 1'b1;
            reset <= 1'b1;
			end	    
        else if (((cnt_initial >= 28'd50_000_000)&&(cnt_write <= 6'd32))&&(cnt_write_times < 5'd20))
			begin
			cnt_initial <= 28'd50_000_000;
			state  <= 3'b010 ;
            SPISTE <= 1'b0 ;
            pdn <= 1'b1;
            end
        else if (((cnt_initial >= 28'd50_000_000)&&(cnt_write > 6'd32))&&(cnt_write_times < 5'd20))
            begin
            state  <= 3'b011 ;
            SPISTE <= 1'b1 ;
            pdn <= 1'b1;
            end
        else if((cnt_initial >= 28'd50_000_000)&&(cnt_write_times >= 5'd20))
            begin
            if(cnt_interval <= 16'd50_000)
                begin
                cnt_interval <= cnt_interval + 1'b1 ;
                state  <= 3'b100 ;
                SPISTE <= 1'b1 ;
                end
            else if((cnt_read <= 6'd32)&&(cnt_interval > 16'd50_000))
                  begin
                cnt_interval <= 16'd50_001 ;
                state  <= 3'b101 ;
                SPISTE <= 1'b0 ;
                    end
            else if((cnt_read > 6'd32)&&(cnt_interval > 16'd50_000))
                begin
                cnt_interval <= 16'd50_001 ;
                state  <= 3'b110 ;
                SPISTE <= 1'b1 ;
                end 
            end
            end
            
           
  /*    
//状态机设置
//参数为50_000_
always@ (posedge clk or negedge rst_n)
	begin
		if(rst_n == 1'b0)
			begin
			state <= 3'b0 ;
            cnt_initial <= 0;
            SPISTE <= 1'b1 ;
			end
		else if(cnt_initial < 28'd50_000_)
			begin
			state  <= 3'b001 ;
			cnt_initial <= cnt_initial + 1'b1 ;
            SPISTE <= 1'b1 ;
			end	
        else if ((cnt_initial >= 28'd50_000_)&&(cnt_write <= 6'd32))
			begin
			cnt_initial <= 28'd50_000_;
			state  <= 3'b010 ;
            SPISTE <= 1'b0 ;
            end
        else if ((cnt_initial >= 28'd50_000_)&&(cnt_write > 6'd32))
            begin
            state  <= 3'b011 ;
            SPISTE <= 1'b1 ;
            end
		else if ((cnt_initial >= 28'd50_000_000)&&(cnt_data_write <= 8'd41)&&(cnt_write <= 6'd33))
			begin
			cnt_initial <= 28'd50_000_000;
			state  <= 3'b010 ;
            SPISTE <= 1'b0 ;
            end
        else if ((cnt_initial >= 28'd50_000_000)&&(cnt_data_write <= 8'd41)&&(cnt_write > 6'd33))
            begin
            state  <= 3'b011 ;
            SPISTE <= 1'b1 ;
            end
        else if((cnt_initial >= 28'd50_000_000)&&(cnt_data_write == 8'd42))
            begin
            if(cnt_interval <= 16'd50_000)
                begin
                cnt_interval <= cnt_interval + 1'b1 ;
                state  <= 3'b100 ;
                SPISTE <= 1'b1 ;
                end
            else if(cnt_read <= 6'd33)
                  begin
                cnt_interval <= 16'd50_001 ;
                state  <= 3'b101 ;
                SPISTE <= 1'b0 ;
                    end
            else if(cnt_read > 6'd33)
                begin
                cnt_interval <= 16'd50_001 ;
                state  <= 3'b110 ;
                SPISTE <= 1'b1 ;
                end 
            end
            end */          
            
            

/*
//reset信号
always@ (posedge clk )
	begin
	if((state == 3'b001)||(state == 3'b000))
		reset <= 1'b0;
	else 
		reset <= 1'b1;
	end
	*/
//SCLK_tmp信号生成
always@ (posedge clk or negedge rst_n)
	begin
		if(rst_n == 1'b0)
			sclk_tmp <= 1'b0 ;
		else if(cnt_sclk < 3'd5)
			cnt_sclk <= cnt_sclk + 1'b1;
		else if(cnt_sclk == 3'd5)
			begin
				sclk_tmp <= ~sclk_tmp ;
				cnt_sclk <= 3'b0 ;
			end
	end

//cnt_tmp
/*    always@ (posedge clk )
	begin
		if(state == 3'b000)
			cnt_tmp <= 1'b0 ;
		else if(cnt_tmp < 4'd11)
			cnt_tmp <= cnt_tmp + 1'b1;
		else if(cnt_tmp == 4'd11)
				cnt_tmp <= 3'b0 ;
	end
	*/
//
always@ (posedge sclk_tmp or negedge rst_n)
	begin
        if(rst_n == 1'b0)
            cnt_write = 6'b1;
        else if((state == 3'b010)||(state == 3'b011))
            cnt_write = cnt_write + 1'b1 ;
        else
            cnt_write = 6'b1;

	end
    
always@ (negedge clk )
    begin
       /* if(((state == 3'b010)||(state == 3'b101))&&(cnt_write != 6'h20))
            sclk <= sclk_tmp ;*/
				if(state == 3'b0)
				sclk <= 1'b0 ;
			/*	else if((state == 3'b010)||(state == 3'b101)) 
                    begin
                    if((cnt_write != 6'h00)&&(cnt_write != 6'h21))
                        sclk <= sclk_tmp ;
                    else
                        sclk <= 1'b0 ;
                    end
                else if((state == 3'b101)||(state == 3'b110)) 
                    begin
                    if((cnt_read != 6'h00)&&(cnt_read != 6'h21))
                        sclk <= sclk_tmp ;
                    else
                        sclk <= 1'b0 ;
                    end*/
                else if(((cnt_write != 6'h00)&&(cnt_write != 6'h21))&&((cnt_read != 6'h00)&&(cnt_read != 6'h21)))
                        sclk <= sclk_tmp ;
                    else
                        sclk <= 1'b0 ;
    end
    

    
always@(posedge cnt_write[5] or negedge rst_n)
    begin
        if(rst_n == 1'b0)
        begin
            cnt_data_write <= 8'd0 ;
            cnt_write_times <= 5'd0 ;
        end
        else if(cnt_write[5] == 1'b1)
            //if(state == 3'b010)
                begin
                   if(cnt_data_write <= 8'd41) 
                   cnt_data_write <= cnt_data_write + 1'b1 ;
                   else if((cnt_write_times < 5'd19)&&(cnt_data_write > 8'd41))
                   begin
                   cnt_data_write <= 8'd0 ;
                   cnt_write_times <= cnt_write_times + 1'b1;
                   end
                   else if((cnt_write_times >= 5'd19)&&(cnt_data_write > 8'd41))
                   begin
                   cnt_data_write <= 8'd0 ;
                   cnt_write_times <= 5'd20 ;
                   end
                end            
    end

always@(posedge sclk)
    begin
    if(state == 3'b0)
            data_tmp <= 32'd0 ;
    else
        begin
    case(cnt_data_write)
    8'h00:data_tmp <= W_register_00;
    8'h01:data_tmp <= W_register_01;
    8'h02:data_tmp <= W_register_02;
    8'h03:data_tmp <= W_register_03;  
    8'h04:data_tmp <= W_register_04;
    8'h05:data_tmp <= W_register_05;  
    8'h06:data_tmp <= W_register_06;
    8'h07:data_tmp <= W_register_07;  
    8'h08:data_tmp <= W_register_08;
    8'h09:data_tmp <= W_register_09;  
    8'h0A:data_tmp <= W_register_0A;
    8'h0B:data_tmp <= W_register_0B;  
    8'h0C:data_tmp <= W_register_0C;
    8'h0D:data_tmp <= W_register_0D;  
    8'h0E:data_tmp <= W_register_0E;
    8'h0F:data_tmp <= W_register_0F;  
    8'h10:data_tmp <= W_register_10;
    8'h11:data_tmp <= W_register_11;
    8'h12:data_tmp <= W_register_12;
    8'h13:data_tmp <= W_register_13;  
    8'h14:data_tmp <= W_register_14;
    8'h15:data_tmp <= W_register_15;  
    8'h16:data_tmp <= W_register_16;
    8'h17:data_tmp <= W_register_17;  
    8'h18:data_tmp <= W_register_18;
    8'h19:data_tmp <= W_register_19;  
    8'h1A:data_tmp <= W_register_1A;
    8'h1B:data_tmp <= W_register_1B;  
    8'h1C:data_tmp <= W_register_1C;
    8'h1D:data_tmp <= W_register_1D;  
    8'h1E:data_tmp <= W_register_1E;
    8'h1F:data_tmp <= W_register_1F; 
    8'h20:data_tmp <= W_register_20;
    8'h21:data_tmp <= W_register_21;
    8'h22:data_tmp <= W_register_22;
    8'h23:data_tmp <= W_register_23;  
    8'h24:data_tmp <= W_register_24;
    8'h25:data_tmp <= W_register_25;  
    8'h26:data_tmp <= W_register_26;
    8'h27:data_tmp <= W_register_27;  
    8'h28:data_tmp <= W_register_28;
    8'h29:data_tmp <= W_register_29;  
    default:;
    endcase
        end
    end
    

always@(posedge sclk_tmp or negedge rst_n)
    begin
    if(rst_n == 1'b0)
        cnt_read <= 6'b1 ;
    else if((state == 3'b101)||(state == 3'b110))
        cnt_read <= cnt_read + 1'b1 ;
    else
        cnt_read <= 6'b1 ;
    end 

always@(posedge cnt_read[5] or negedge AFE_RDY_N)
    begin
        if(AFE_RDY_N == 1'b0)
            cnt_address_read <= 8'd0 ;
       // else if(cnt_read[5] == 1'b1)
        else if((cnt_address_read <= 8'h6)&&(AFE_RDY_N == 1'b1))
               
                  // if(((cnt_address_read <= 8'd48) && (cnt_address_read >= 8'd42))||(cnt_address_read == 8'd0))
					
						//if((cnt_address_read == 8'd0)||((cnt_address_read <= 8'd48) && (cnt_address_read >= 8'd42)))
                   cnt_address_read <= cnt_address_read + 1'b1 ;
         else
         cnt_address_read <= 8'd8;
                          
    end
    
always@(posedge sclk_tmp or negedge rst_n)
    begin
    if(rst_n == 1'b0)
            addr_read_tmp <= 6'b0 ;   
    else 
    begin case(cnt_address_read)
        8'b0 :addr_read_tmp <= 8'h00;
        8'h1:addr_read_tmp <= 8'h2A;
        8'h2:addr_read_tmp <= 8'h2B;
        8'h3:addr_read_tmp <= 8'h2C;
        8'h4:addr_read_tmp <= 8'h2D;
        8'h5:addr_read_tmp <= 8'h2E;
        8'h6:addr_read_tmp <= 8'h2F;
        8'h7:addr_read_tmp <= 8'h30;
        default:addr_read_tmp <= 8'h00;
    endcase 
    end
    end
    
always@(negedge sclk or posedge SPISTE) 

       begin
        if(SPISTE == 1'b1)
            SPISIMO <= 1'b0 ; 
        else if(state == 3'b010)
        begin
               case(cnt_write)
                6'd01:SPISIMO <= data_tmp[30];
                6'd02:SPISIMO <= data_tmp[29];
                6'd03:SPISIMO <= data_tmp[28];
                6'd04:SPISIMO <= data_tmp[27];
                6'd05:SPISIMO <= data_tmp[26];
                6'd06:SPISIMO <= data_tmp[25];
                6'd07:SPISIMO <= data_tmp[24];
                6'd08:SPISIMO <= data_tmp[23];
                6'd09:SPISIMO <= data_tmp[22];
                6'd10:SPISIMO <= data_tmp[21];
                6'd11:SPISIMO <= data_tmp[20];
                6'd12:SPISIMO <= data_tmp[19];
                6'd13:SPISIMO <= data_tmp[18];
                6'd14:SPISIMO <= data_tmp[17];
                6'd15:SPISIMO <= data_tmp[16];
                6'd16:SPISIMO <= data_tmp[15];
                6'd17:SPISIMO <= data_tmp[14];
                6'd18:SPISIMO <= data_tmp[13];
                6'd19:SPISIMO <= data_tmp[12];
                6'd20:SPISIMO <= data_tmp[11];
                6'd21:SPISIMO <= data_tmp[10];
                6'd22:SPISIMO <= data_tmp[9] ;
                6'd23:SPISIMO <= data_tmp[8] ;
                6'd24:SPISIMO <= data_tmp[7] ;
                6'd25:SPISIMO <= data_tmp[6] ;
                6'd26:SPISIMO <= data_tmp[5] ;
                6'd27:SPISIMO <= data_tmp[4] ;
                6'd28:SPISIMO <= data_tmp[3] ;
                6'd29:SPISIMO <= data_tmp[2] ;
                6'd30:SPISIMO <= data_tmp[1] ;
                6'd31:SPISIMO <= data_tmp[0] ;
                6'd32:SPISIMO <= data_tmp[0] ;
                default:SPISIMO <= 1'b0;
                endcase
        end
        else if((addr_read_tmp == 8'b0)&&(state == 3'b101))
        begin
            case(cnt_read)
            6'd01:SPISIMO <= addr_read_tmp[6];
            6'd02:SPISIMO <= addr_read_tmp[5];
            6'd03:SPISIMO <= addr_read_tmp[4];
            6'd04:SPISIMO <= addr_read_tmp[3];
            6'd05:SPISIMO <= addr_read_tmp[2];
            6'd06:SPISIMO <= addr_read_tmp[1];
            6'd07:SPISIMO <= addr_read_tmp[0];
            6'd08:SPISIMO <= 1'b0;
            6'd09:SPISIMO <= 1'b0;
            6'd10:SPISIMO <= 1'b0;
            6'd11:SPISIMO <= 1'b0;
            6'd12:SPISIMO <= 1'b0;
            6'd13:SPISIMO <= 1'b0;
            6'd14:SPISIMO <= 1'b0;
            6'd15:SPISIMO <= 1'b0;
            6'd16:SPISIMO <= 1'b0;
            6'd17:SPISIMO <= 1'b0;
            6'd18:SPISIMO <= 1'b0;
            6'd19:SPISIMO <= 1'b0;
            6'd20:SPISIMO <= 1'b0;
            6'd21:SPISIMO <= 1'b0;
            6'd22:SPISIMO <= 1'b0;
            6'd23:SPISIMO <= 1'b0;
            6'd24:SPISIMO <= 1'b0;
            6'd25:SPISIMO <= 1'b0;
            6'd26:SPISIMO <= 1'b0;
            6'd27:SPISIMO <= 1'b0;
            6'd28:SPISIMO <= 1'b0;
            6'd29:SPISIMO <= 1'b0;
            6'd30:SPISIMO <= 1'b0;
            6'd31:SPISIMO <= 1'b1;
            6'd32:SPISIMO <= 1'b0;
            default:SPISIMO <= 1'b0;
            endcase
        end
        else if((addr_read_tmp != 8'b0)&&(state == 3'b101))
        begin
            case(cnt_read)
            6'd01:SPISIMO <= addr_read_tmp[6];
            6'd02:SPISIMO <= addr_read_tmp[5];
            6'd03:SPISIMO <= addr_read_tmp[4];
            6'd04:SPISIMO <= addr_read_tmp[3];
            6'd05:SPISIMO <= addr_read_tmp[2];
            6'd06:SPISIMO <= addr_read_tmp[1];
            6'd07:SPISIMO <= addr_read_tmp[0];
            6'd08:begin SPISIMO <= 1'b0;data_read_tmp[23] <= 1'b0 ; end
            6'd09:begin SPISIMO <= 1'b0;data_read_tmp[22] <= 1'b0 ; end
            6'd10:begin SPISIMO <= 1'b0;data_read_tmp[21] <= SPISOMI ; end
            6'd11:begin SPISIMO <= 1'b0;data_read_tmp[20] <= SPISOMI ; end
            6'd12:begin SPISIMO <= 1'b0;data_read_tmp[19] <= SPISOMI ; end
            6'd13:begin SPISIMO <= 1'b0;data_read_tmp[18] <= SPISOMI ; end
            6'd14:begin SPISIMO <= 1'b0;data_read_tmp[17] <= SPISOMI ; end
            6'd15:begin SPISIMO <= 1'b0;data_read_tmp[16] <= SPISOMI ; end
            6'd16:begin SPISIMO <= 1'b0;data_read_tmp[15] <= SPISOMI ; end
            6'd17:begin SPISIMO <= 1'b0;data_read_tmp[14] <= SPISOMI ; end
            6'd18:begin SPISIMO <= 1'b0;data_read_tmp[13] <= SPISOMI ; end
            6'd19:begin SPISIMO <= 1'b0;data_read_tmp[12] <= SPISOMI ; end
            6'd20:begin SPISIMO <= 1'b0;data_read_tmp[11] <= SPISOMI ; end
            6'd21:begin SPISIMO <= 1'b0;data_read_tmp[10] <= SPISOMI ; end
            6'd22:begin SPISIMO <= 1'b0;data_read_tmp[9] <= SPISOMI ; end
            6'd23:begin SPISIMO <= 1'b0;data_read_tmp[8] <= SPISOMI ; end
            6'd24:begin SPISIMO <= 1'b0;data_read_tmp[7] <= SPISOMI ; end
            6'd25:begin SPISIMO <= 1'b0;data_read_tmp[6] <= SPISOMI ; end
            6'd26:begin SPISIMO <= 1'b0;data_read_tmp[5] <= SPISOMI ; end
            6'd27:begin SPISIMO <= 1'b0;data_read_tmp[4] <= SPISOMI ; end
            6'd28:begin SPISIMO <= 1'b0;data_read_tmp[3] <= SPISOMI ; end
            6'd29:begin SPISIMO <= 1'b0;data_read_tmp[2] <= SPISOMI ; end
            6'd30:begin SPISIMO <= 1'b0;data_read_tmp[1] <= SPISOMI ; end
            6'd31:begin SPISIMO <= 1'b0;data_read_tmp[0] <= SPISOMI ; end
            6'd32:begin SPISIMO <= 1'b0; end
            default:SPISIMO <= 1'b0;
            endcase
        end
        
     end 

always@(posedge sclk_tmp)
    begin
        case(addr_read_tmp)
        8'b0 :       ;
        8'h2A:       ;
        8'h2B:       ;
        8'h2C:       ;
        8'h2D:       ;
        8'h2E:  LED2_substracted <= data_read_tmp    ;
        8'h2F:  LED1_substracted <= data_read_tmp    ;
        8'h30:       ;
        default:;
        endcase
		//  data_out <= {8'h2E,LED2_substracted,8'h2F,LED1_substracted};
    end
    
always@(negedge AFE_RDY)
    begin
    data_out <= {LED2_substracted,LED1_substracted};
    end 


     
endmodule
